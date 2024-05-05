import torch

class ContentEmb(torch.nn.Module):
    """
    in content (canny) 
    """
    def __init__(self):
        super().__init__()
        self.inconv = torch.nn.Sequential(
            torch.nn.Conv2d(1,30, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(30),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(30,30, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(30),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(30,77, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(77),
            torch.nn.ReLU(inplace=True),
        )

        # out linear
        self.dense_out = torch.nn.Sequential(
            torch.nn.Linear(4096, 1024),
            torch.nn.LeakyReLU(0.2,inplace=True),

            torch.nn.Linear(1024, 1024),
            torch.nn.LeakyReLU(0.2,inplace=True),

            torch.nn.Linear(1024, 768),
            torch.nn.LeakyReLU(0.2,inplace=True),
        )
        
    
    def forward(self, content_image):
        """
        [B,1,512,512]-> [B,77,768]
        """

        # [B,1,512,512] -> [B,77,64,64]
        out_encoder = self.inconv(content_image)
        
        # [B,77,64,64] -> [B,77,64*64]
        B,C,H,W = out_encoder.size()
        out_encoder = out_encoder.reshape(B, C, H*W)

        # out linear [B,77,64*64]-> [B,77,768]
        out_encoder = self.dense_out(out_encoder)

        return out_encoder
    