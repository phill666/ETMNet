import torch
import torch.nn as nn


class EdgeFlowConv(nn.Module):
    """Directional edge-flow convolution for boundary-sensitive detail features."""

    def __init__(self, in_channels, out_channels=None, reduction=4):
        super().__init__()
        out_channels = in_channels if out_channels is None else int(out_channels)
        hidden_channels = max(8, int(in_channels) // int(reduction))
        self.in_channels = int(in_channels)
        self.out_channels = out_channels

        self.compress = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_channels, 4, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.expand = nn.Conv2d(hidden_channels * 4, self.out_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(self.out_channels)
        self.shortcut = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x_shrink = self.compress(x)
        gate = self.gate(x_shrink)

        feat_h = self.conv_h(x_shrink) * gate[:, 0:1]
        feat_v = self.conv_v(x_shrink) * gate[:, 1:2]
        feat_d1 = self.conv_d1(x_shrink) * gate[:, 2:3]
        feat_d2 = self.conv_d2(x_shrink) * gate[:, 3:4]

        out = self.expand(torch.cat([feat_h, feat_v, feat_d1, feat_d2], dim=1))
        out = self.expand_bn(out)
        return out + self.shortcut(x)

    def reset_residual(self):
        nn.init.zeros_(self.expand.weight)
        nn.init.ones_(self.expand_bn.weight)
        nn.init.zeros_(self.expand_bn.bias)

