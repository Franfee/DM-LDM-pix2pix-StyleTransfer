# -*- coding: utf-8 -*-
# @Time    : 2023/12/6
# @Author  : FanAnfei
# @python  : Python 3.9.12

import torch
from net.UNet import DownBlock, PositionalEmbedding, Resnet, Transformer
from net.VGG import extract_features


class StyleFusionCondition(torch.nn.Module):
    def __init__(self):
        """
        LIST -> [B, 320, 64, 64]
        """
        super().__init__()
        self.fit0 = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),              # [1, 64, 512, 512] -> [1, 64, 256, 256]
            torch.nn.Conv2d(64, 120, 3, 2, 1),  # [1, 64, 256, 256] -> [1, 120, 128, 128]
            torch.nn.BatchNorm2d(120),
            torch.nn.SiLU(),

            torch.nn.Conv2d(120, 320, 3, 2, 1),  # [1, 120, 128, 128] -> [1, 320, 64, 64]
            torch.nn.BatchNorm2d(320),
            torch.nn.SiLU()
        )
        self.gama0 = torch.nn.Parameter(torch.Tensor([0.2]))

        self.fit1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, 2, 1), # [1, 128, 256, 256] -> [1, 256, 128, 128]
            torch.nn.BatchNorm2d(256),
            torch.nn.SiLU(),

            torch.nn.Conv2d(256, 320, 3, 2, 1), # [1, 256, 128, 128] -> [1, 320, 64, 64]
            torch.nn.BatchNorm2d(320),
            torch.nn.SiLU()
        )
        self.gama1 = torch.nn.Parameter(torch.Tensor([0.2]))

        self.fit2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, 2, 1), # [1, 256, 128, 128] -> [1, 256, 64, 64]
            torch.nn.BatchNorm2d(256),
            torch.nn.SiLU(),

            torch.nn.Conv2d(256, 320, 3, 1, 1), # [1, 256, 64, 64] -> [1, 320, 64, 64]
            torch.nn.BatchNorm2d(320),
            torch.nn.SiLU()
        )
        self.gama2 = torch.nn.Parameter(torch.Tensor([0.2]))

        self.fit3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 320, 3, 1, 1), # [1, 512, 64, 64] -> [1, 320, 64, 64]
            torch.nn.BatchNorm2d(320),
            torch.nn.SiLU()
        )
        self.gama3 = torch.nn.Parameter(torch.Tensor([0.2]))

        self.fit4 = torch.nn.Sequential(
            torch.nn.PixelShuffle(upscale_factor=2), # [1, 512, 32, 32] -> [1, 128, 64, 64]
            torch.nn.BatchNorm2d(128),
            torch.nn.SiLU(),

            torch.nn.Conv2d(128, 320, 3, 1, 1), # [1, 128, 64, 64] -> [1, 320, 64, 64]
            torch.nn.BatchNorm2d(320),
            torch.nn.SiLU()
        )
        self.gama4 = torch.nn.Parameter(torch.Tensor([0.2]))


    def forward(self, condition):
        """
        @condition: [1, 64, 512, 512],[1, 128, 256, 256],[1, 256, 128, 128],[1, 512, 64, 64],[1, 512, 32, 32]
        return [1, 320, 64, 64]
        """
        assert len(condition) == 5

        #return torch.zeros(1,320,64,64)
        f0 = self.fit0(condition[0])
        f1 = self.fit1(condition[1])
        f2 = self.fit2(condition[2])
        f3 = self.fit3(condition[3])
        f4 = self.fit4(condition[4])
        return self.gama0 * f0 + self.gama1 * f1 + self.gama2 * f2 + self.gama3 * f3 + self.gama4 * f4
        
class StyleEmb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        入参embed部分
        分别对图像,噪声步数,草图数据进行投影
        """

        # self.in_vae
        self.in_vae = torch.nn.Conv2d(4, 320, kernel_size=3, padding=1)
        
        # scale(timesteps)
        self.time_embed = PositionalEmbedding(num_channels=192)
        self.in_time = torch.nn.Sequential(
            torch.nn.Linear(192, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280),
        )

        # style control
        self.condition_embed = StyleFusionCondition()

        # unet的down部分
        self.down_block0 = DownBlock(320, 320)
        self.down_block1 = DownBlock(320, 640)
        self.down_block2 = DownBlock(640, 1280)

        self.down_res0 = Resnet(1280, 1280)
        self.down_res1 = Resnet(1280, 1280)

        # unet的mid部分
        self.mid_res0 = Resnet(1280, 1280)
        self.mid_tf = Transformer(1280)
        self.mid_res1 = Resnet(1280, 1280)

        """
        control的down部分
        """
        self.control_down = torch.nn.ModuleList([
            torch.nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(640, 640, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(1280, 1280, kernel_size=1, stride=1, padding=0),
        ])

        """
        control的mid部分
        """
        self.control_mid = torch.nn.Conv2d(1280, 1280, kernel_size=1)
    
    
    def forward(self, out_vae_noise, out_encoder, noise_step, condition):
        """
        计算部分
        计算过程,两张图片分别进行投影

        @out_vae_noise -> [1, 4, 64, 64]  :VAE编码之后的特征矩阵
        @noise_step -> [1]

        @out_encoder -> [1, 77 ,768]      :内容指导
        @condition -> LIST len = 5  风格指导
        """

        # [1] -> [1, 192]
        noise_step = self.time_embed(noise_step)
        # [1, 320] -> [1, 1280]
        noise_step = self.in_time(noise_step)

        # vae详细图编码out_vae_noise经过投影升到高维
        # [1, 4, 64, 64] -> [1, 320, 64, 64]
        out_vae_noise = self.in_vae(out_vae_noise)

        # 风格列表投影到和out_vae_noise同一维度空间
        # [LIST] -> [1, 320, 64, 64]
        condition = self.condition_embed(condition)

        # 向out_vae_noise中添加风格condition信息,得到混合图编码
        # [1, 320, 64, 64]
        out_vae_noise += condition

        # unet的down部分计算(unet有4次down),每一层当中包括了3个串行的注意力计算,所以每一层都有3个计算结果.
        # [1, 320, 64, 64]
        # [1, 320, 64, 64]
        # [1, 320, 64, 64]
        # [1, 320, 32, 32]
        # [1, 640, 32, 32]
        # [1, 640, 32, 32]
        # [1, 640, 16, 16]
        # [1, 1280, 16, 16]
        # [1, 1280, 16, 16]
        # [1, 1280, 8, 8]
        # [1, 1280, 8, 8]
        # [1, 1280, 8, 8]
        out_unet_down = [out_vae_noise]

        # [1, 320, 64, 64],[1, 77, 768],[1, 1280] -> [1, 320, 32, 32]
        # out -> [1, 320, 64, 64],[1, 320, 64, 64][1, 320, 32, 32]
        out_vae_noise, out = self.down_block0(out_vae_noise=out_vae_noise, out_encoder=out_encoder, time=noise_step)
        out_unet_down.extend(out)

        # [1, 320, 32, 32],[1, 77, 768],[1, 1280] -> [1, 640, 16, 16]
        # out -> [1, 640, 32, 32],[1, 640, 32, 32],[1, 640, 16, 16]
        out_vae_noise, out = self.down_block1(out_vae_noise=out_vae_noise, out_encoder=out_encoder, time=noise_step)
        out_unet_down.extend(out)

        # [1, 640, 16, 16],[1, 77, 768],[1, 1280] -> [1, 1280, 8, 8]
        # out -> [1, 1280, 16, 16],[1, 1280, 16, 16],[1, 1280, 8, 8]
        out_vae_noise, out = self.down_block2(out_vae_noise=out_vae_noise, out_encoder=out_encoder, time=noise_step)
        out_unet_down.extend(out)

        # [1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae_noise = self.down_res0(out_vae_noise, noise_step)
        out_unet_down.append(out_vae_noise)

        # [1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae_noise = self.down_res1(out_vae_noise, noise_step)
        out_unet_down.append(out_vae_noise)



        # unet的mid计算,维度不变
        # [1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae_noise = self.mid_res0(out_vae_noise, noise_step)

        # [1, 1280, 8, 8],[1, 77, 768] -> [1, 1280, 8, 8]
        out_vae_noise = self.mid_tf(out_vae_noise, out_encoder)

        # [1, 1280, 8, 8],[1, 1280] -> [1, 1280, 8, 8]
        out_vae_noise = self.mid_res1(out_vae_noise, noise_step)

        # control的down的部分计算,维度不变,两两组合,分别计算即可
        out_control_down = [
            self.control_down[i](out_unet_down[i]) for i in range(12)
        ]

        # control的mid的部分计算,维度不变
        out_control_mid = self.control_mid(out_vae_noise)

        return out_control_down, out_control_mid



if __name__ == '__main__':
    se = StyleEmb()

    se.requires_grad_(False)

    out_vae = torch.randn(1,4,64,64)
    out_con = torch.randn(1,77,768)
    timescale = torch.Tensor([0.1])
    
    cond = torch.randn(1,3,512,512)

    # [[1, 512, 64, 64]] 
    # [[1, 64, 512, 512],[1, 128, 256, 256],[1, 512, 64, 64],[1, 512, 32, 32]]
    cond_c, cond_s = extract_features(cond)


    o = se(out_vae, out_con, timescale, cond_s)
    print(o.shape)

