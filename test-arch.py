import torch
from net.EDMPrecond import EDMPrecond
from net.ContentEmb import ContentEmb
from net.StyleEmb import StyleEmb
from net.UNet import UNet
from net.VAE import VAE
from net.VGG import extract_features
from training.loss import EDMLoss


diffusion = UNet()
diffusion.requires_grad_(False)

edm = EDMPrecond(model=diffusion)

vae1 = VAE()
vae1.eval()
vae1.requires_grad_(False)

styleEmbControl = StyleEmb()
styleEmbControl.eval()
styleEmbControl.requires_grad_(False)

contentEncoder = ContentEmb()
contentEncoder.eval()
contentEncoder.requires_grad_(False)


# dataloader item: 内容图像 z，内容edge图像, 风格图像
img_c = torch.randn(1,3,512,512)
img_e = torch.randn(1,1,512,512)
img_s = torch.randn(1,3,512,512)


# 潜空间
z = vae1.sample(vae1.encoder(img_c))

print("laten z:", z.shape)

# 内容指导 嵌入空间
c_emb = contentEncoder(img_e)
print("content emb:", c_emb.shape) 


# 去噪隐空间
out_vae = torch.rand_like(z)
scale = torch.Tensor([0.01])


# 内容，风格提取
_c, s = extract_features(img_s)
print("style seq:", len(s))

# 风格指导 嵌入空间
down_block_additional_residuals, mid_block_additional_residual= styleEmbControl(out_vae_noise=out_vae, 
                                                                                out_encoder=c_emb, 
                                                                                noise_step=scale, 
                                                                                condition=s)
for dd in down_block_additional_residuals:
    print("\tdown res shape:", dd.shape)
print("\tmid res shape:", mid_block_additional_residual.shape)


# -----------------------------------
lossFun = EDMLoss()

# loss 训练stage1
loss = lossFun(net=edm, y=z, condition=(c_emb, None))
# loss 训练stage2
loss = lossFun(net=edm, y=z, condition=(c_emb, (down_block_additional_residuals, mid_block_additional_residual)))

loss_scaling = 1 
t = 256

# 梯度传播
# loss.sum().mul(loss_scaling / t).backward()

print("loss:", loss.sum().mul(loss_scaling / t).item())

# -----------test stage-------------
# 内容指导去噪
out_vae_denoised = diffusion(out_vae_noise=out_vae, out_encoder=c_emb, time=scale, 
                     down_block_additional_residuals=None, mid_block_additional_residual=None)
print("out_vae_denoised:", out_vae_denoised.shape)

# 还原图像
img_rec = vae1.decoder(out_vae_denoised)
print("img rec:", img_rec.shape)