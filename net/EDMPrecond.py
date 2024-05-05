import torch

class EDMPrecond(torch.nn.Module):
    def __init__(self,
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model           = None,
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, x, sigma, condition, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)

        (content, style) = condition
        if style is not None:
            (down_block_additional_residuals, mid_block_additional_residual) = style
        else:
            down_block_additional_residuals=None
            mid_block_additional_residual=None

        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model(out_vae_noise=(c_in * x).to(dtype), 
                         time=c_noise.flatten(), 
                         out_encoder=content, 
                         down_block_additional_residuals=down_block_additional_residuals,
                         mid_block_additional_residual=mid_block_additional_residual)
        
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)