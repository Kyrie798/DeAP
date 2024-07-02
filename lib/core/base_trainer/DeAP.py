import torch
import torch.nn as nn

from lib.core.base_trainer.MCFM import MCFM
from lib.core.base_trainer.Stripformer import Stripformer

# Learning Degradation-Aware Prior
class DeAP(nn.Module):
    def __init__(self):
        super(DeAP, self).__init__()
        
        # Stripformer
        self.backbone = Stripformer()

        # Momentum Contrast Feature Module
        self.MCFM = MCFM()

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(128, 320, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        )
    
    def forward(self, x_query, x_key):
        if self.training:
            inter, logits, labels = self.MCFM(x_query, x_key)
            inter_down1 = self.down1(inter)
            inter_down2 = self.down2(inter_down1)

            restored = self.backbone(x_query, inter, inter_down1, inter_down2)

            return restored, logits, labels
        else:
            inter = self.MCFM(x_query, x_key)
            inter_down1 = self.down1(inter)
            inter_down2 = self.down2(inter_down1)

            restored = self.backbone(x_query, inter, inter_down1, inter_down2)

            return restored