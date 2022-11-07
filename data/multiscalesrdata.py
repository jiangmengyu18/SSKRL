import os
import glob

from data import common
import pickle
import numpy as np
import imageio

import torch
import torch.utils.data as data


class SRData(data.Dataset):
    def __init__(self, args, benchmark=False):
        self.args = args
        self.benchmark = benchmark
        if self.benchmark:
            self.name = args.data_test
        else:
            self.name = args.data_train
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)
        self.list_hr = self._scan()
        self.images_hr = []
        if self.benchmark:
            self.images_hr = self.list_hr
        else:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            for h in self.list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check([h], b, verbose=True)

        if not self.benchmark:
            self.repeat = args.test_every // (len(self.images_hr) // args.batch_size)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )

        return names_hr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.ext = ('.png', '.png')

    def _check(self, l, f, verbose=True):
        if not os.path.isfile(f):
            if verbose:         
                print('{} does not exist. Now making binary...'.format(f))
            b = [{
                'name': os.path.splitext(os.path.basename(_l))[0],
                'image': imageio.imread(_l)
            } for _l in l]
            with open(f, 'wb') as _f:
                pickle.dump(b, _f)

    def __getitem__(self, idx):
        hr, filename = self._load_file(idx)
        hr = self.get_patch(hr)
        hr = [common.set_channel(img, n_channels=self.args.n_colors) for img in hr]
        hr_tensor = [common.np2Tensor(img, rgb_range=self.args.rgb_range)
                     for img in hr]
        return torch.stack(hr_tensor, 0), filename

    def __len__(self):
        if not self.benchmark:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if not self.benchmark:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.benchmark:
            hr = imageio.imread(f_hr)
        else:
            with open(f_hr, 'rb') as _f:
                hr = np.load(_f, allow_pickle=True)[0]['image']

        return hr, filename

    def get_patch(self, hr):
        scale = self.scale[self.idx_scale]
        if not self.benchmark:
            out = []
            hr = common.augment(hr) if not self.args.no_augment else hr
            # extract two patches from each image
            for _ in range(2):
                hr_patch = common.get_patch(
                    hr,
                    patch_size=self.args.patch_size,
                    scale=scale
                )
                out.append(hr_patch)
        else:
            out = [hr]
        return out

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale


class HRLRData(data.Dataset):
    def __init__(self, args, benchmark=True):
        self.args = args
        self.benchmark = benchmark
        self.name = args.data_test
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)
        self.images_hr, self.images_lr, self.images_k = self._scan()

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])),
            key=lambda arr: int(arr.split('_')[1]),
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '*' + self.ext[0])),
            key=lambda arr: int(arr.split('_')[2].split('.')[0])
        )
        names_k = sorted(
            glob.glob(os.path.join(self.dir_k, '*' + self.ext[0]))
        )

        return names_hr, names_lr, names_k

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_x' + str(int(self.scale[self.idx_scale])))
        self.dir_k = os.path.join(self.apath, 'k_x' + str(int(self.scale[self.idx_scale])))
        self.ext = ('.png', '.png')

    def __getitem__(self, idx):
        hr, lr, k = self._load_file(idx)
    
        hr = common.set_channel(hr, n_channels=self.args.n_colors)
        hr_tensor = common.np2Tensor(hr, rgb_range=self.args.rgb_range)

        lr = common.set_channel(lr, n_channels=self.args.n_colors)
        lr_tensor = common.np2Tensor(lr, rgb_range=self.args.rgb_range)

        k_tensor = torch.from_numpy(k).float()
                     
        return hr_tensor, lr_tensor, k_tensor

    def __len__(self):
        return len(self.images_hr)

    def _load_file(self, idx):
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[idx]
        f_k = self.images_k[idx]

        hr = imageio.imread(f_hr)
        lr = imageio.imread(f_lr)
        k = imageio.imread(f_k)

        return hr, lr, k

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale