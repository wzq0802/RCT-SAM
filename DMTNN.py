import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity
        return F.relu(out)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        layers = [ResidualBlock(in_channels, out_channels)]
        for _ in range(3):
            layers.append(ResidualBlock(out_channels, out_channels))
        self.block = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.block(x)
        pooled = self.pool(x)
        return x, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.block = ResidualBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.size()[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class ImprovedLinkNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ImprovedLinkNet, self).__init__()

        # 参数初始化层
        self.init = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),

            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # 编码器
        self.encoder1 = EncoderBlock(64, 128)
        self.encoder2 = EncoderBlock(128, 256)
        self.encoder3 = EncoderBlock(256, 512)

        # bottleneck
        self.bottleneck = ResidualBlock(512, 1024)

        # 解码器
        self.decoder3 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder1 = DecoderBlock(256, 128)

        # 最终上采样（恢复到原图大小）
        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),  # 64→256
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.init(x)  # 256→64
        skip1, x = self.encoder1(x)  # 64→32
        skip2, x = self.encoder2(x)  # 32→16
        skip3, x = self.encoder3(x)  # 16→8
        x = self.bottleneck(x)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        out = self.up_final(x)  # 64→256
        return out

#
# # 测试
# if __name__ == "__main__":
#     model = ImprovedLinkNet()
#     x = torch.randn(1, 1, 256, 256)
#     y = model(x)
#     print("Output shape:", y.shape)

