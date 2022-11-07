import os
import utility
import torch
import numpy as np
from decimal import Decimal
import torch.nn.functional as F
from utils import util
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.model_E = self.model.get_model().E.cuda()
        self.sr_loss = torch.nn.L1Loss().cuda()
        self.kernel_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = torch.nn.CosineSimilarity(dim=1).cuda()
        self.L1 = torch.nn.L1Loss().cuda()
        self.optimizer = utility.make_optimizer(args, self.model)
        self.epoch = 0

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

    def train(self):
        epoch = self.epoch + 1

        # lr stepwise
        if epoch <= self.args.epochs_encoder:
            lr = self.args.lr_encoder * (self.args.gamma_encoder ** (epoch // self.args.lr_decay_encoder))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self.args.lr_sr * (self.args.gamma_sr ** ((epoch - self.args.epochs_encoder) // self.args.lr_decay_sr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.model.train()

        degrade = util.SRMDPreprocessing(
            self.scale[0],
            rgb_range=self.args.rgb_range,
            mode=self.args.mode,
            kernel_size=self.args.blur_kernel,
            blur_type=self.args.blur_type,
            sig_min=self.args.sig_min,
            sig_max=self.args.sig_max,
            lambda_min=self.args.lambda_min,
            lambda_max=self.args.lambda_max,
            noise=self.args.noise,
        )

        timer = utility.timer()
        losses_kernel, losses_contrast, losses_sr = utility.AverageMeter(), utility.AverageMeter(), utility.AverageMeter()

        for batch, (hr, filename) in tqdm(enumerate(self.loader_train)):
            hr_0 = hr[:, 0, :, :, :]                    # b, c, h, w
            hr_1 = hr[:, 1, :, :, :]                    # b, c, h, w
            hr_1 = hr_1[torch.randperm(hr_1.size(0))]   # b, c, h, w
            hr = torch.stack([hr_0, hr_1], 1)           # b, n, c, h, w
            hr = hr.cuda()                              # b, n, c, h, w
            lr, b_kernels = degrade(hr)                 # b, n, c, h, w
            torch.set_printoptions(threshold=np.inf)
            self.optimizer.zero_grad()

            timer.tic()
            # forward
            ## train degradation encoder
            if epoch <= self.args.epochs_encoder:
                p0, p1, z0, z1, k0, k1 = self.model_E(x0=lr[:,0,...], x1=lr[:,1,...])
                loss_constrast = -self.args.weight_cl * (self.contrast_loss(p0, z1.detach()).mean() + self.contrast_loss(p1, z0.detach()).mean()) * 0.5
                loss_kernel = self.args.weight_kl * (self.kernel_loss(k0, b_kernels.flatten(1)) + self.kernel_loss(k1, b_kernels.flatten(1))) * 0.5
                loss = loss_kernel + loss_constrast

                losses_contrast.update(loss_constrast.item())
                losses_kernel.update(loss_kernel.item())
            ## train the whole network
            else:
                sr0, sr1, p0, p1, z0, z1, k0, k1 = self.model(lr)
                loss_SR = self.args.weight_srl * (self.sr_loss(sr0, hr[:,0,...]) + self.sr_loss(sr1, hr[:,1,...])) * 0.5
                loss_constrast = -self.args.weight_cl * (self.contrast_loss(p0, z1.detach()).mean() + self.contrast_loss(p1, z0.detach()).mean()) * 0.5
                loss_kernel = self.args.weight_kl * (self.kernel_loss(k0, b_kernels.flatten(1)) + self.kernel_loss(k1, b_kernels.flatten(1))) * 0.5
                loss = loss_constrast + loss_SR + loss_kernel

                losses_sr.update(loss_SR.item())
                losses_kernel.update(loss_kernel.item())
                losses_contrast.update(loss_constrast.item())

            # backward
            loss.backward()
            self.optimizer.step()
            timer.hold()

            if epoch <= self.args.epochs_encoder:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:03d}][{:04d}/{:04d}]\t'
                        'Loss [kernel loss: {:.5f} | contrastive loss: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_kernel.avg, losses_contrast.avg,
                            timer.release()
                        ))
            else:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                        'Loss [SR loss:{:.3f} | kernel loss:{:.5f} | contrastive loss: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_sr.avg, losses_kernel.avg, losses_contrast.avg,
                            timer.release(),
                        ))

        if epoch > 300:
            # save model
            target = self.model.get_model()
            model_dict = target.state_dict()
            keys = list(model_dict.keys())
            for key in keys:
                if 'E.encoder_k' in key or 'queue' in key:
                    del model_dict[key]
            torch.save(
                model_dict,
                os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
            )
        self.epoch += 1
        self.args.resume = self.epoch

    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)

                if self.args.HRLR:
                    eval_psnr = 0
                    eval_ssim = 0
                    eval_K_mse = 0
                    for idx_img, (hr, lr, b_kernels) in tqdm(enumerate(self.loader_test)):
                        hr = hr.cuda()                      # b, c, h, w
                        lr = lr.cuda()                      # b, c, h, w
                        b_kernels = b_kernels.cuda()        # b, s, s

                        # inference
                        timer_test.tic()
                        sr, k = self.model(lr)
                        timer_test.hold()

                        # metrics
                        eval_psnr += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        eval_ssim += utility.calc_ssim(
                            sr * (255 / self.args.rgb_range), hr * (255 / self.args.rgb_range), scale,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        eval_K_mse += self.L1(k, b_kernels.flatten(1)) * 1000
                    
                    self.ckp.write_log(
                        '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} K_L1: {:.4f}'.format(
                            self.args.resume,
                            self.args.data_test,
                            scale,
                            eval_psnr / len(self.loader_test),
                            eval_ssim / len(self.loader_test),
                            eval_K_mse / len(self.loader_test),
                    ))
                else:
                    if self.args.blur_type == "iso_gaussian":
                        sigs = [float(item) for item in self.args.sig.split(',')]
                        eval_psnr_all = []
                        eval_ssim_all = []
                        eval_K_mse_all = []
                        for sig in sigs:
                            degrade = util.SRMDPreprocessing(
                                self.scale[0],
                                rgb_range=self.args.rgb_range,
                                mode=self.args.mode,
                                kernel_size=self.args.blur_kernel,
                                blur_type=self.args.blur_type,
                                sig=sig,
                            )

                            eval_psnr = 0
                            eval_ssim = 0
                            eval_K_mse = 0
                            for idx_img, (hr, filename) in enumerate(self.loader_test):
                                hr = hr.cuda()                      # b, 1, c, h, w
                                hr = self.crop_border(hr, scale)
                                lr, b_kernels = degrade(hr, random=False)   # b, 1, c, h, w
                                hr = hr[:, 0, ...]                  # b, c, h, w

                                # inference
                                timer_test.tic()
                                sr, k = self.model(lr[:, 0, ...])
                                timer_test.hold()

                                # metrics
                                eval_psnr += utility.calc_psnr(
                                    sr, hr, scale, self.args.rgb_range,
                                    benchmark=self.loader_test.dataset.benchmark
                                )
                                eval_ssim += utility.calc_ssim(
                                    sr * (255 / self.args.rgb_range), hr * (255 / self.args.rgb_range), scale,
                                    benchmark=self.loader_test.dataset.benchmark
                                )
                                eval_K_mse += self.L1(k, b_kernels.flatten(1)) * 1000

                                # save results
                                if self.args.save_results:
                                    save_list = [sr]
                                    filename = filename[0]
                                    self.ckp.save_results(filename, save_list, scale)
                            
                            eval_psnr_all.append(eval_psnr / len(self.loader_test))
                            eval_ssim_all.append(eval_ssim / len(self.loader_test))
                            eval_K_mse_all.append(eval_K_mse / len(self.loader_test))

                        self.ckp.write_log(
                            '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} K_L1: {:.4f}'.format(
                                self.args.resume,
                                self.args.data_test,
                                scale,
                                sum(eval_psnr_all) / len(eval_psnr_all),
                                sum(eval_ssim_all) / len(eval_ssim_all),
                                sum(eval_K_mse_all) / len(eval_K_mse_all),
                        ))
                    elif self.args.blur_type == "aniso_gaussian":
                        lambda_1s = [float(item) for item in self.args.lambda_1.split(',')]
                        lambda_2s = [float(item) for item in self.args.lambda_2.split(',')]
                        thetas = [float(item) for item in self.args.theta.split(',')]
                        eval_psnr_all = []
                        eval_ssim_all = []
                        eval_K_mse_all = []
                        for index in range(len(lambda_1s)):
                            degrade = util.SRMDPreprocessing(
                                self.scale[0],
                                rgb_range=self.args.rgb_range,
                                mode=self.args.mode,
                                kernel_size=self.args.blur_kernel,
                                blur_type=self.args.blur_type,
                                lambda_1=lambda_1s[index],
                                lambda_2=lambda_2s[index],
                                theta=thetas[index],
                                noise=self.args.noise_test,
                            )

                            eval_psnr = 0
                            eval_ssim = 0
                            eval_K_mse = 0
                            for idx_img, (hr, filename) in enumerate(self.loader_test):
                                hr = hr.cuda()                      # b, 1, c, h, w
                                hr = self.crop_border(hr, scale)
                                lr, b_kernels = degrade(hr, random=False)   # b, 1, c, h, w
                                hr = hr[:, 0, ...]                  # b, c, h, w

                                # inference
                                timer_test.tic()
                                sr, k = self.model(lr[:, 0, ...])
                                timer_test.hold()

                                # metrics
                                eval_psnr += utility.calc_psnr(
                                    sr, hr, scale, self.args.rgb_range,
                                    benchmark=self.loader_test.dataset.benchmark
                                )
                                eval_ssim += utility.calc_ssim(
                                    sr * (255 / self.args.rgb_range), hr * (255 / self.args.rgb_range), scale,
                                    benchmark=self.loader_test.dataset.benchmark
                                )
                                eval_K_mse += self.L1(k, b_kernels.flatten(1)) * 1000

                                # save results
                                if self.args.save_results:
                                    save_list = [sr]
                                    filename = filename[0]
                                    self.ckp.save_results(filename, save_list, scale)
                            
                            eval_psnr_all.append(eval_psnr / len(self.loader_test))
                            eval_ssim_all.append(eval_ssim / len(self.loader_test))
                            eval_K_mse_all.append(eval_K_mse / len(self.loader_test))

                        self.ckp.write_log(
                            '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} K_L1: {:.4f}'.format(
                                self.args.resume,
                                self.args.data_test,
                                scale,
                                sum(eval_psnr_all) / len(eval_psnr_all),
                                sum(eval_ssim_all) / len(eval_ssim_all),
                                sum(eval_K_mse_all) / len(eval_K_mse_all),
                        ))
    def crop_border(self, img_hr, scale):
        b, n, c, h, w = img_hr.size()

        img_hr = img_hr[:, :, :, :int(h//scale*scale), :int(w//scale*scale)]

        return img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            return self.epoch >= self.args.epochs_encoder + self.args.epochs_sr

