import torch
from torch import nn
import model.common as common
import torch.nn.functional as F
from moco.builder import SimSiam


def make_model(args):
    return BlindSR(args)


class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = common.default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C * 1 * 1
        '''
        b, c, h, w = x[0].size()

        # branch 1
        kernel = self.kernel(x[1].squeeze(-1).squeeze(-1)).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        out = self.conv(out.view(b, -1, h, w))

        # branch 2
        out = out + self.ca(x)

        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C * 1 * 1
        '''
        att = self.conv_du(x[1])

        return x[0] * att


class DPKB(nn.Module):
    def __init__(self, nf1, nf2, ksize1=3, ksize2=1):
        super().__init__()

        self.body0 = nn.Sequential(
            nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2),
        )

        self.body1 = nn.Sequential(
            nn.Conv2d(nf2, nf2, ksize2, 1, ksize2 // 2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf2, nf2, ksize2, 1, ksize2 // 2),
        )

        self.body2 = DA_conv(nf1, nf1, 3, 8)

        self.conv_du1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nf1, nf1 // 8, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf1 // 8, nf1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.conv_du2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nf2, nf2 // 8, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf2 // 8, nf2, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        f0 = self.body0(x[0])
        f1 = self.body1(x[1])
        
        f0 = self.body2([f0, f1])
        
        x[0] = x[0] + f0 * self.conv_du1(f0)
        x[1] = x[1] + f1 * self.conv_du2(f1)
        
        return x


class DPKG(nn.Module):
    def __init__(self, nf1, nf2, ksize1, ksize2, nb):
        super().__init__()

        self.body = nn.Sequential(*[DPKB(nf1, nf2, ksize1, ksize2) for _ in range(nb)])

    def forward(self, x):
        y = self.body(x)
        y[0] = x[0] + y[0]
        y[1] = x[1] + y[1]
        return y


class DKSR(nn.Module):
    def __init__(self, args):
        super(DKSR, self).__init__()
        in_nc = 3
        nf = 64
        ng = 5
        nb = 10
        scale = int(args.scale[0])
        out_nc = in_nc

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head1 = nn.Conv2d(in_nc, nf, 3, stride=1, padding=1)
        self.head2 = nn.Conv2d(256, nf, 1, 1, 0)

        body = [DPKG(nf, nf, 3, 1, nb) for _ in range(ng)]
        self.body = nn.Sequential(*body)

        self.fusion = nn.Conv2d(nf, nf, 3, 1, 1)

        if scale == 4:  # x4
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale // 2),
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )
        elif scale == 1:
            self.upscale = nn.Conv2d(nf, out_nc, 3, 1, 1)

        else:  # x2, x3
            self.upscale = nn.Sequential(
                nn.Conv2d(
                    in_channels=nf,
                    out_channels=nf * scale ** 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.PixelShuffle(scale),
                nn.Conv2d(nf, out_nc, 3, 1, 1),
            )

    def forward(self, input, ker_code):
        B, C, H, W = input.size()  # I_LR batch
        B_h, C_h = ker_code.size()  # Batch, Len=10
        ker_code_exp = ker_code.view((B_h, C_h, 1, 1))

        # sub mean
        input = self.sub_mean(input)

        f1 = self.head1(input)
        f2 = self.head2(ker_code_exp)
        inputs = [f1, f2]
        f, _ = self.body(inputs)
        f = self.fusion(f)
        out = self.upscale(f)

        # add mean
        out = self.add_mean(out)

        return out


class Tail(nn.Module):
    def __init__(self, args):
        super(Tail, self).__init__()

        self.mlp_c = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, True), # hidden layer
            nn.Linear(256, 256) # output layer
        ) 
        self.mlp_k = nn.Sequential(
            nn.Linear(256, args.blur_kernel * args.blur_kernel, bias=False),
            nn.BatchNorm1d(args.blur_kernel * args.blur_kernel),
            nn.LeakyReLU(0.1, True), # hidden layer
            nn.Linear(args.blur_kernel * args.blur_kernel, args.blur_kernel * args.blur_kernel), # output layer
            nn.Softmax(1)
        )
    
    def forward(self, x):
        out_c = self.mlp_c(x)
        out_k = self.mlp_k(x)
        
        return out_c, out_k


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.T = Tail(args)

    def forward(self, x):
        z = self.E(x).squeeze(-1).squeeze(-1)
        p, k = self.T(z)

        return z, p, k


class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        self.G = DKSR(args)

        # Encoder
        self.E = SimSiam(args=args, base_encoder=Encoder)

    def forward(self, x):
        if self.training:
            x0 = x[:, 0, ...]                          # b, c, h, w
            x1 = x[:, 1, ...]                            # b, c, h, w

            # degradation-aware represenetion learning
            p0, p1, z0, z1, k0, k1 = self.E(x0, x1)

            # degradation-aware SR
            sr0 = self.G(x0, z0)
            sr1 = self.G(x1, z1)

            return sr0, sr1, p0, p1, z0, z1, k0, k1
        else:
            # degradation-aware represenetion learning
            z, k = self.E(x, x)

            # degradation-aware SR
            sr = self.G(x, z)

            return sr, k
