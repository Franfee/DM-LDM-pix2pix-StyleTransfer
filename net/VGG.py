# -*- coding: utf-8 -*-
# @Time    : 2022/12/12 10:20
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12

import torch
from torch import nn
import torchvision

# 预训练path
LOAD_PATH = "net/pretrained/vgg19-dcbb9e9d.pth"

#
STYLE_LAYERS, CONTENT_LAYERS = [0, 5, 10, 19, 28], [25]


print("loading vgg19-dcbb9e9d...")
pretrained_net = torchvision.models.vgg19()
pretrained_net.load_state_dict(torch.load(LOAD_PATH))
print("loaded vgg19-dcbb9e9d!")

# 为了抽取图像的内容特征和⻛格特征，我们可以选择VGG⽹络中某些层的输出。⼀般来说，越靠近输⼊层，越 容易抽取图像的细节信息；
# 反之，则越容易抽取图像的全局信息。为了避免合成图像过多保留内容图像的细节，我们选择VGG较靠近输出的层，即内容层，来输出图像的内容特征。
# 我们还从VGG中选择不同层的输出来匹配局部和全局的⻛格，这些图层也称为⻛格层。VGG⽹络使⽤了5个卷积块。
# 实验中，我们选择第四卷积块的最后⼀个卷积层作为内容层，选择每个卷积块的第⼀个卷积层作为⻛格层。
# 这些层的索引可以通过打印pretrained_net实例获取。

# 使⽤VGG层抽取特征时，我们只需要⽤到从输⼊层到最靠近输出层的内容层或⻛格层之间的所有层。
# 下⾯构建⼀个新的⽹络net，它只保留需要⽤到的VGG的所有层。
VGG19 = nn.Sequential(*[pretrained_net.features[i] for i in range(max(CONTENT_LAYERS + STYLE_LAYERS) + 1)])
VGG19.requires_grad_(False)
del pretrained_net


@torch.no_grad()
def extract_features(images):
    """
    给定输入 images, 如果我们调用前向传播net(images)，只能获得最后一层的输出。 由于我们还需要中间层的输出，因此这里我们逐层计算，并保留内容层和风格层的输出。
    :param images:
    :return:
    """
    contents = []
    styles = []
    for i in range(len(VGG19)):
        images = VGG19[i](images)
        if i in STYLE_LAYERS:
            styles.append(images)
        if i in CONTENT_LAYERS:
            contents.append(images)
    return contents, styles


if __name__ == '__main__':
    img = torch.randn(1,3,512,512)


    c, s = extract_features(img)

    for cc in c:
        print(cc.shape)
    
    for ss in s:
        print(ss.shape)