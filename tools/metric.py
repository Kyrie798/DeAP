import numpy as np
import torch.nn as nn

from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

class PSNRMeter(nn.Module):
    def __init__(self):
        super().__init__()

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 0.5) * 255.0
        return image_numpy

    def cal_psnr(self, output, target):
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)
        psnr = PSNR(fake, real, data_range=255)
        ssim = SSIM(fake.astype('uint8'), real.astype('uint8'), channel_axis=2)
        return psnr, ssim