"""
Pyramid ViG-Ti model definition.
Adapted from the official ViG implementation:
https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class MRConv2d(nn.Module):
    """Max-Relative Graph Convolution (used in ViG)."""

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU() if act == 'relu' else nn.GELU()

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = nn.BatchNorm2d(out_channels) if norm == 'bn' else nn.Identity()

    def forward(self, x, edge_index):
        # x: (N, C), edge_index: (2, E)
        row, col = edge_index
        x_j = x[row] - x[col]  # max-relative
        x_j = F.relu(x_j) if isinstance(self.act, nn.ReLU) else self.act(x_j)
        out = torch.zeros_like(x)
        out.index_add_(0, col, x_j)
        out = self.fc(out)
        return self.norm(out)


class Grapher(nn.Module):
    """
    Grapher module: graph construction (kNN) + graph convolution + FFN.
    This is the core building block of ViG.
    """

    def __init__(self, in_channels, kernel_size=9, dilation=1, drop_path=0.0):
        super().__init__()
        self.k = kernel_size  # k for kNN
        self.dilation = dilation
        self.channels = in_channels

        # Graph convolution (max-relative)
        self.graph_conv = MRConv2d(in_channels, in_channels * 2, act='relu', norm='bn')

        # FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 4, 1, bias=False),
            nn.BatchNorm2d(in_channels * 4),
            nn.GELU(),
            nn.Conv2d(in_channels * 4, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _build_knn_graph(self, x):
        """
        Build kNN graph from node features.
        x: (B, C, H, W) -> flatten to (B*H*W, C)
        Returns edge_index for each batch (concatenated).
        """
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)  # (B, N, C)

        # Compute pairwise cosine similarity (batched)
        x_norm = F.normalize(x_flat, dim=-1)
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))  # (B, N, N)

        # Top-k neighbors (excluding self)
        sim.diagonal(dim1=1, dim2=2).fill_(-float('inf'))
        _, knn_idx = sim.topk(self.k, dim=-1)  # (B, N, k)

        # Build edge_index
        row = torch.arange(N, device=x.device).view(1, N, 1).expand(B, -1, self.k)
        edge_index = torch.stack([row.reshape(-1), knn_idx.reshape(-1)], dim=0)  # (2, B*N*k)
        return edge_index, B, N

    def forward(self, x):
        B, C, H, W = x.shape
        # --- Graph Construction ---
        edge_index, B, N = self._build_knn_graph(x)

        # --- Graph Convolution ---
        x_flat = x.permute(0, 2, 3, 1).reshape(B * N, C)
        x_conv = self.graph_conv(x_flat, edge_index)
        x_conv = x_conv.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # --- FFN ---
        x_ffn = self.ffn(x_conv)

        # --- Residual ---
        x = self.drop_path(x_ffn) + x
        return x


class PyramidViG(nn.Module):
    """
    Pyramid ViG-Ti (tiny) architecture.
    Stages: 4 stages with depths [2, 2, 6, 2] as in the original paper.
    """

    def __init__(self, num_classes=1000, in_chans=3, embed_dim=48,
                 depths=(2, 2, 6, 2), drop_path_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # Stem: patch embedding (16x16 conv, stride 16)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=16, stride=16, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # Stages
        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i, depth in enumerate(depths):
            dim = embed_dim * (2 ** i)
            if i > 0:
                # Downsample between stages
                downsample = nn.Sequential(
                    nn.Conv2d(embed_dim * (2 ** (i - 1)), dim, kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm2d(dim),
                )
            else:
                downsample = nn.Identity()

            stage = nn.Sequential(
                downsample,
                *[Grapher(dim, kernel_size=9, drop_path=dpr[cur + j])
                  for j in range(depth)]
            )
            self.stages.append(stage)
            cur += depth

        # Classification head
        self.norm = nn.BatchNorm2d(embed_dim * 8)
        self.head = nn.Linear(embed_dim * 8, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.norm(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        x = self.head(x)
        return x


def pyramid_vig_ti(num_classes=1000):
    """Factory function for Pyramid ViG-Ti."""
    return PyramidViG(
        num_classes=num_classes,
        embed_dim=48,
        depths=(2, 2, 6, 2),
        drop_path_rate=0.1,
    )