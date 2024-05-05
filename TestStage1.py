
import os
from PIL import Image
import torch

from net.ContentEmb import ContentEmb
from net.EDMPrecond import EDMPrecond
from net.UNet import UNet
from net.VAE import VAE
from sample.edm_sampler import edm_sampler
from training.get_dataLoader_contentA import get_data_loader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


DEPOCH = 1
CEPOCH = 1


#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])
    
def save_samples(images, batch_seeds, out_dir):
    images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for seed, image_np in zip(batch_seeds, images_np):
        image_dir = os.path.join(out_dir, f'{seed - seed % 1000:06d}')
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f'{seed:06d}.png')
        if image_np.shape[2] == 1:
            Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
        else:
            Image.fromarray(image_np, 'RGB').save(image_path)


#----------------------------------------------------------------------------


def main():
    vae = VAE()
    vae.LoadPreTrain()
    vae.to(DEVICE)
    vae.eval()
    vae.requires_grad_(False)
    
    # 扩散模型
    diffusion = UNet()
    diffusion.load_state_dict(torch.load(os.path.join("result", "model", f"{DEPOCH}.diffusion.ckpt"), map_location="cpu"))

    edm = EDMPrecond(model=diffusion)
    edm.to(DEVICE)
    
    # 内容模块
    contentEncoder = ContentEmb()
    contentEncoder.load_state_dict(torch.load(os.path.join("result", "model", f"{CEPOCH}.contentEncoder.ckpt"), map_location="cpu"))
    contentEncoder.to(DEVICE)

    # eval
    edm.eval()
    edm.requires_grad_(False)
    contentEncoder.eval()
    contentEncoder.requires_grad_(False)

    dataset_iterator = iter(get_data_loader(mode="test", bathsize=1))


    for batch_seeds in range(999):
        data = next(dataset_iterator)
        gt, content = data['lantent'].to(DEVICE), data['content'].to(DEVICE)

        # seed
        rnd = StackedRandomGenerator(DEVICE, batch_seeds)
        latents = rnd.randn([1, 4, 64, 64], device=DEVICE)
        
        # 内容条件
        c_emb = contentEncoder(content)
        de = edm_sampler(net=edm, latents=latents, condition=(c_emb,None))

        gt_img = vae.decoder(gt)
        de_img = vae.decoder(de)
        save_samples(gt_img, batch_seeds, "result"+"/gt512")
        save_samples(de_img, batch_seeds, "result"+"/de512")


if __name__ == "__main__":
    main()