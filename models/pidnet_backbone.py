import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DetailBranch(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNReLU(in_channels, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.stage3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return self.stage3(x)


class StemBlock(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        x = self.conv(x)
        left = self.left(x)
        right = self.right(x)
        return self.fuse(torch.cat([left, right], dim=1))


class CEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        gap = torch.mean(x, dim=(2, 3), keepdim=True)
        if self.training and gap.numel() // gap.shape[1] == 1:
            gap = self.conv_gap(gap)
        else:
            gap = self.conv_gap(self.bn(gap))
        return self.conv_last(x + gap)


class GELayerS1(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=6):
        super().__init__()
        mid_channels = in_channels * expansion
        self.conv1 = ConvBNReLU(in_channels, in_channels, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dwconv(out)
        out = self.conv2(out)
        return self.relu(out + x)


class GELayerS2(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=6):
        super().__init__()
        mid_channels = in_channels * expansion
        self.conv1 = ConvBNReLU(in_channels, in_channels, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dwconv1(out)
        out = self.dwconv2(out)
        out = self.conv2(out)
        return self.relu(out + self.shortcut(x))


class SegmentBranch(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.stage12 = StemBlock(in_channels)
        self.stage3 = nn.Sequential(GELayerS2(16, 32), GELayerS1(32, 32))
        self.stage4 = nn.Sequential(GELayerS2(32, 64), GELayerS1(64, 64))
        self.stage5_4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.stage5_5 = CEBlock()

    def forward(self, x):
        feat2 = self.stage12(x)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        feat5_4 = self.stage5_4(feat4)
        feat5_5 = self.stage5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, bias=False),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, detail_feat, segment_feat):
        dsize = detail_feat.shape[2:]
        left1 = self.left1(detail_feat)
        left2 = self.left2(detail_feat)
        right1 = self.right1(segment_feat)
        right2 = self.right2(segment_feat)

        right1 = F.interpolate(right1, size=dsize, mode="bilinear", align_corners=False)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = F.interpolate(right, size=dsize, mode="bilinear", align_corners=False)
        return self.conv(left + right)


class SegmentHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, mid_channels, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

    def forward(self, x, out_size):
        x = self.conv(x)
        x = self.drop(x)
        x = self.conv_out(x)
        if x.shape[2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x

