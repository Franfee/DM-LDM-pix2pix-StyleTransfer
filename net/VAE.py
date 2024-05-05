# -*- coding: utf-8 -*-
# @Time : 2023/4/25 21:18
# @Author : FanAnfei
# @File : VAE.py
# @Software: PyCharm
import os

import torch


class Resnet(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.s = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=dim_in, eps=1e-6, affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_groups=32, num_channels=dim_out, eps=1e-6, affine=True),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
        )

        self.res = None
        if dim_in != dim_out:
            self.res = torch.nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        x -> [B, 128, 10, 10]
        :param x:
        :return:
        """

        res = x
        if self.res:
            # [B, 128, 10, 10] -> [B, 256, 10, 10]
            res = self.res(x)

        # [B, 128, 10, 10] -> [B, 256, 10, 10]
        return res + self.s(x)


class SelfAtten(torch.nn.Module):

    def __init__(self):
        """
        自注意力，无mask
        """
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6, affine=True)

        self.q = torch.nn.Linear(512, 512)
        self.k = torch.nn.Linear(512, 512)
        self.v = torch.nn.Linear(512, 512)
        self.out = torch.nn.Linear(512, 512)

    def forward(self, x):
        """
        x -> [B, 512, 64, 64]
        :param x:
        :return:
        """

        res = x

        # norm,维度不变
        # [B, 512, 64, 64]
        x = self.norm(x)

        # [B, 512, 64, 64] -> [B, 512, 4096] -> [B, 4096, 512]
        x = x.flatten(start_dim=2).transpose(1, 2)

        # 线性运算,维度不变
        # [B, 4096, 512]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # [B, 4096, 512] -> [B, 512, 4096]
        k = k.transpose(1, 2)

        # [B, 4096, 512] * [B, 512, 4096] -> [B, 4096, 4096]
        # 0.044194173824159216 = 1 / 512**0.5
        # atten = q.bmm(k) * 0.044194173824159216

        # 照理来说应该是等价的,但是却有很小的误差
        atten = torch.baddbmm(torch.empty(1, 4096, 4096, device=q.device), q, k, beta=0, alpha=0.044194173824159216)
        atten = torch.softmax(atten, dim=2)

        # [B, 4096, 4096] * [B, 4096, 512] -> [B, 4096, 512]
        atten = atten.bmm(v)

        # 线性运算,维度不变
        # [B, 4096, 512]
        atten = self.out(atten)

        # [B, 4096, 512] -> [B, 512, 4096] -> [B, 512, 64, 64]
        atten = atten.transpose(1, 2).reshape(-1, 512, 64, 64)

        # 残差连接,维度不变
        # [B, 512, 64, 64]
        atten = atten + res

        return atten


class Pad(torch.nn.Module):

    def forward(self, x):
        return torch.nn.functional.pad(x, (0, 1, 0, 1), mode='constant', value=0)


class VAE(torch.nn.Module):
    """
        图片编码解码模型
    """
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            # in
            torch.nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),

            # down
            torch.nn.Sequential(
                Resnet(128, 128),
                Resnet(128, 128),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(128, 256),
                Resnet(256, 256),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(256, 512),
                Resnet(512, 512),
                torch.nn.Sequential(
                    Pad(),
                    torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
                ),
            ),
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
            ),

            # mid
            torch.nn.Sequential(
                Resnet(512, 512),
                SelfAtten(),
                Resnet(512, 512),
            ),

            # out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6),
                torch.nn.SiLU(),
                torch.nn.Conv2d(512, 8, kernel_size=3, padding=1),
            ),

            # 正态分布层
            torch.nn.Conv2d(8, 8, kernel_size=1),
        )

        self.decoder = torch.nn.Sequential(
            # 正态分布层
            torch.nn.Conv2d(4, 4, 1),

            # in
            torch.nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1),

            # middle
            torch.nn.Sequential(Resnet(512, 512), SelfAtten(), Resnet(512, 512)),

            # up
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(512, 512),
                Resnet(512, 512),
                Resnet(512, 512),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(512, 256),
                Resnet(256, 256),
                Resnet(256, 256),
                torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
                torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ),
            torch.nn.Sequential(
                Resnet(256, 128),
                Resnet(128, 128),
                Resnet(128, 128),
            ),

            # out
            torch.nn.Sequential(
                torch.nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6),
                torch.nn.SiLU(),
                torch.nn.Conv2d(128, 3, kernel_size=3, padding=1),
            ),
        )

    def sample(self, h):
        """
        h -> [1, 8, 64, 64]
        :param h:
        :return:
        """
        # [B, 4, 64, 64]
        mean = h[:, :4]
        logvar = h[:, 4:]
        std = logvar.exp() ** 0.5

        # [B, 4, 64, 64]
        h = torch.randn(mean.shape, device=mean.device)
        h = mean + std * h

        return h

    def forward(self, x):
        """
        编码->投影->解码
        x -> [B, 3, 512, 512]
        :param x:
        :return:
        """

        # [B, 3, 512, 512] -> [B, 8, 64, 64]
        h = self.encoder(x)

        # [B, 8, 64, 64] -> [B, 4, 64, 64]
        h = self.sample(h)

        # [B, 4, 64, 64] -> [B, 3, 512, 512]
        h = self.decoder(h)

        return h

    def LoadPreTrain(self):
        BASE_DIR = os.getcwd()
        if "net" not in BASE_DIR:
            BASE_DIR = os.path.join(BASE_DIR, "net")
        FILE_DIR = os.path.join(BASE_DIR, "pretrained", "VAE.pt")
        self.load_state_dict(torch.load(FILE_DIR))
        print("VAE loaded from a local copy.")


def Test():
    print("resnet test:")
    out = Resnet(128, 256)(torch.randn(1, 128, 10, 10))
    print(out.shape)

    print("atten test:")
    out = SelfAtten()(torch.randn(1, 512, 64, 64))
    print(out.shape)

    print("VAE test:")
    out = VAE()(torch.randn(1, 3, 512, 512))
    print(out.shape)


def Test2():
    vae1 = VAE()
    print("load param test(local_copy):")
    vae1.LoadPreTrain()

    data = torch.randn(1, 3, 512, 512)

    h1 = vae1.encoder(data)
    print("h1:", h1.shape)
    s1 = vae1.sample(h1)
    print("s1:", s1.shape)
    o1 = vae1.decoder(s1)
    print("o1:", o1.shape)


if __name__ == '__main__':
    Test()
    Test2()

    # RuntimeError: The expanded size of the tensor (3072) must match the existing size (4096) at non-singleton dimension 2.  
    # Target sizes: [1, 3072, 3072].  Tensor sizes: [1, 4096, 4096]
    # vae1 = VAE()
    # data = torch.randn(1, 3, 384, 512)

    # h1 = vae1.encoder(data)
    # print("h1:", h1.shape)
    # s1 = vae1.sample(h1)
    # print("s1:", s1.shape)
    # o1 = vae1.decoder(s1)
    # print("o1:", o1.shape)
