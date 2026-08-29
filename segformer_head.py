# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Linear Embedding 模块
    将输入的 2D 特征展平后通过线性层映射维度
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    纯 PyTorch 实现的 SegFormer All-MLP 解码器
    """
    def __init__(
        self,
        in_channels=[32, 64, 160, 256],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=1,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        **kwargs
    ):
        super(SegFormerHead, self).__init__()
        assert len(feature_strides) == len(in_channels)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.align_corners = align_corners

        c1_in, c2_in, c3_in, c4_in = self.in_channels
        embedding_dim = decoder_params['embed_dim']

        # 分别对 4 个 stage 的特征进行线性映射
        self.linear_c4 = MLP(input_dim=c4_in, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in, embed_dim=embedding_dim)

        # 替代 mmcv 的 ConvModule (Conv2d + BatchNorm2d + ReLU)
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        # Dropout 与最终预测分类头
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        """
        inputs: Backbone 4 个 stage 输出的特征列表 [c1, c2, c3, c4]
        """
        c1, c2, c3, c4 = inputs
        n = c4.shape[0]

        # Stage 4 MLP + 上采样至 c1 尺寸 (H/4, W/4)
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        # Stage 3 MLP + 上采样至 c1 尺寸
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        # Stage 2 MLP + 上采样至 c1 尺寸
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        # Stage 1 MLP
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # 4 尺度特征通道拼接并融合
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x