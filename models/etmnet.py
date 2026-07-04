import torch
import torch.nn as nn
import torch.nn.functional as F

from .edgeflowconv import EdgeFlowConv
from .pidnet_backbone import BGALayer, ConvBNReLU, DetailBranch, SegmentBranch, SegmentHead


class LocalCrossChannelAttention(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_feat = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        max_feat = self.max_pool(x).squeeze(-1).transpose(-1, -2)
        avg_attn = self.conv1d(avg_feat).transpose(-1, -2).unsqueeze(-1)
        max_attn = self.conv1d(max_feat).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(avg_attn + max_attn)


class GradientAwareSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion_conv = nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_x = torch.mean(grad_x, dim=1, keepdim=True)

        grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        grad_y = torch.mean(grad_y, dim=1, keepdim=True)

        attn = self.sigmoid(self.fusion_conv(torch.cat([avg_out, max_out, grad_x, grad_y], dim=1)))
        return x * attn


class AdaptiveReceptiveRouting(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = (
            nn.Identity()
            if int(in_channels) == int(out_channels)
            else nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=False)
        )
        out_channels = int(out_channels)
        self.branch_d1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.branch_d2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.ReLU(inplace=True),
        )
        self.branch_d3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.ReLU(inplace=True),
        )
        mid_channels = max(8, out_channels // 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_channels, mid_channels, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(mid_channels, 3 * out_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.align(x)
        batch, channels, _, _ = x.shape
        feat_1 = self.branch_d1(x)
        feat_2 = self.branch_d2(x)
        feat_3 = self.branch_d3(x)
        pooled = self.gap(feat_1 + feat_2 + feat_3)
        route = self.fc2(self.relu(self.fc1(pooled))).view(batch, 3, channels, 1, 1)
        route = F.softmax(route, dim=1)
        return feat_1 * route[:, 0] + feat_2 * route[:, 1] + feat_3 * route[:, 2]


class TopologyContextBlock(nn.Module):
    """Lightweight context block used on the semantic branch."""

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = in_channels if out_channels is None else int(out_channels)
        self.shortcut = (
            nn.Identity()
            if int(in_channels) == out_channels
            else nn.Conv2d(int(in_channels), out_channels, kernel_size=1, bias=False)
        )
        self.lcca = LocalCrossChannelAttention(int(in_channels))
        self.gasa = GradientAwareSpatialAttention()
        self.arr = AdaptiveReceptiveRouting(int(in_channels), out_channels)
        self.out_norm = nn.BatchNorm2d(out_channels)
        self.out_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.lcca(x)
        x = self.gasa(x)
        x = self.arr(x)
        x = self.out_proj(self.out_norm(x))
        return residual + x

    def reset_residual(self):
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.ones_(self.out_norm.weight)
        nn.init.zeros_(self.out_norm.bias)


class BoundaryHead(nn.Module):
    def __init__(self, in_channels, mid_channels=64):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.head(x)


class ETMNet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        aux_loss=True,
        in_channels=3,
        use_edgeflowconv=True,
        use_topology_context=True,
        use_boundary_aux=True,
        edge_gamma=0.3,
        topology_gamma=0.2,
        **kwargs,
    ):
        super().__init__()
        if "n_classes" in kwargs:
            num_classes = kwargs.pop("n_classes")
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError("Unexpected ETMNet arguments: {}".format(unknown))

        self.num_classes = int(num_classes)
        self.aux_loss = bool(aux_loss)
        self.use_edgeflowconv = bool(use_edgeflowconv)
        self.use_topology_context = bool(use_topology_context)
        self.use_boundary_aux = bool(use_boundary_aux)
        self.edge_gamma = float(edge_gamma)
        self.topology_gamma = float(topology_gamma)

        self.detail = DetailBranch(in_channels)
        self.segment = SegmentBranch(in_channels)
        self.edge_flow = EdgeFlowConv(128, 128) if self.use_edgeflowconv else None
        self.topology_context = TopologyContextBlock(128, 128) if self.use_topology_context else None
        self.bga = BGALayer()
        self.head = SegmentHead(128, 1024, self.num_classes)
        self.boundary_head = BoundaryHead(128) if self.use_boundary_aux else None

        if self.aux_loss:
            self.aux2 = SegmentHead(16, 128, self.num_classes)
            self.aux3 = SegmentHead(32, 128, self.num_classes)
            self.aux4 = SegmentHead(64, 128, self.num_classes)
            self.aux5_4 = SegmentHead(128, 128, self.num_classes)

        self.init_weights()
        self.reset_residual_paths()

    def forward(self, x, return_extra=False):
        out_size = x.shape[2:]

        detail_feat = self.detail(x)
        if self.edge_flow is not None:
            edge_feat = self.edge_flow(detail_feat)
            detail_feat = detail_feat + self.edge_gamma * (edge_feat - detail_feat)

        feat2, feat3, feat4, feat5_4, segment_feat = self.segment(x)
        if self.topology_context is not None:
            routed_feat = self.topology_context(segment_feat)
            segment_feat = segment_feat + self.topology_gamma * (routed_feat - segment_feat)

        fused_feat = self.bga(detail_feat, segment_feat)
        logits = self.head(fused_feat, out_size)

        aux_outputs = ()
        if self.training and self.aux_loss:
            aux_outputs = (
                self.aux2(feat2, out_size),
                self.aux3(feat3, out_size),
                self.aux4(feat4, out_size),
                self.aux5_4(feat5_4, out_size),
            )

        boundary_logits = None
        if self.training and self.boundary_head is not None:
            boundary_logits = self.boundary_head(detail_feat)

        if return_extra:
            return logits, aux_outputs, {"boundary_logits": boundary_logits}
        if self.training and aux_outputs:
            return (logits, *aux_outputs)
        return logits

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if getattr(module, "last_bn", False):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def reset_residual_paths(self):
        if self.edge_flow is not None and hasattr(self.edge_flow, "reset_residual"):
            self.edge_flow.reset_residual()
        if self.topology_context is not None and hasattr(self.topology_context, "reset_residual"):
            self.topology_context.reset_residual()
        if self.boundary_head is not None:
            last = self.boundary_head.head[-1]
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def get_params(self):
        wd_params, nowd_params = [], []
        lr_mul_wd_params, lr_mul_nowd_params = [], []
        head_names = ("head", "aux2", "aux3", "aux4", "aux5_4", "boundary_head")

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_head = name.startswith(head_names)
            if param.dim() == 1:
                (lr_mul_nowd_params if is_head else nowd_params).append(param)
            else:
                (lr_mul_wd_params if is_head else wd_params).append(param)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


