# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 17:00
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.9.12


import glob
import random
import os
from PIL import Image
import cv2

from net.VAE import VAE

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as ttf
import torchvision.transforms as transforms


# Image transformations
# transforms_ = [
#     transforms.Resize(int(512 * 1.12), ttf.InterpolationMode.BICUBIC),
#     transforms.RandomCrop((512, 512)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ]
transforms_ = [
    transforms.Resize(int(512), ttf.InterpolationMode.BICUBIC),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

transforms2 = transforms.Compose([
    transforms.Resize(int(512), ttf.InterpolationMode.BICUBIC),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.vae = VAE()
        self.vae.LoadPreTrain()
        self.vae.to(DEVICE)
        self.vae.eval()
        self.vae.requires_grad_(False)

        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))

    @torch.no_grad()
    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        input_image = cv2.imread(self.files_A[index % len(self.files_A)])
        detected_map = cv2.Canny(input_image, 50, 200)
        canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB)).convert('L')
        canny_map = transforms2(canny_map)
        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A).to(DEVICE).reshape(1,3,512,512)
        item_A = self.vae.sample(self.vae.encoder(item_A)).squeeze(0)

        item_B = self.transform(image_B)
        return {"lantent": item_A, "content": canny_map,"style": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


def get_data_loader(mode="train", bathsize=1):
    if mode == "train":
        train_dataloader = DataLoader(
            ImageDataset("datasets/", transforms_=transforms_, unaligned=True),
            batch_size=bathsize,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        return train_dataloader
    else:
        val_dataloader = DataLoader(
            ImageDataset("datasets/", transforms_=transforms_, unalignedc=True, mode="test"),
            batch_size=bathsize,
            shuffle=False,
            num_workers=0,
        )
        return val_dataloader
