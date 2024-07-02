import torch.nn as nn
from lib.core.base_trainer.moco import MoCo

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.leakrelu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        return self.leakrelu(self.Conv(x) + self.residual(x))

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = BasicBlock(in_channel=3, out_channel=64, stride=1)

        self.backbone = nn.Sequential(
            BasicBlock(in_channel=64, out_channel=128, stride=2),
            BasicBlock(in_channel=128, out_channel=256, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        if self.training:
            inter = self.embedding(x)
            fea = self.backbone(inter).squeeze(-1).squeeze(-1)
            out = self.mlp(fea)
            return inter, out
        else:
            inter = self.embedding(x)
            return inter

class MCFM(nn.Module):
    def __init__(self):
        super(MCFM, self).__init__()
        dim = 256
        self.moco = MoCo(base_encoder=ResNet, dim=dim)

    def forward(self, x_query, x_key):
        if self.training:
            inter, logits, labels = self.moco(x_query, x_key)
            return inter, logits, labels
        else:
            inter = self.moco(x_query, x_key)
            return inter