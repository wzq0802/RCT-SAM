import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# 导入纯 PyTorch 版的 SegFormer 架构
from mix_transformer import mit_b0
from segformer_head import SegFormerHead

# 创建预测图保存目录
save_pred_dir = "SegFormer_Weights/pred"
os.makedirs(save_pred_dir, exist_ok=True)

data_path = os.listdir("data/test/ori-npy")
data_path.sort(key=lambda x: int(x[:-4]))


def default_loader(path):
    data_pil = np.load("data/test/ori-npy/%s" % (path)).reshape((1, 256, 256))
    data_tensor = torch.tensor(data_pil, dtype=torch.float32)
    return data_tensor


class TestSet(Dataset):
    def __init__(self, loader=default_loader):
        self.images = data_path
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        return img

    def __len__(self):
        return len(self.images)

class CompleteSegFormer(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.backbone = mit_b0(in_chans=in_channels)
        self.decode_head = SegFormerHead(
            in_channels=[32, 64, 160, 256],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=0.1,
            num_classes=num_classes,
            align_corners=False,
            decoder_params=dict(embed_dim=256)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        features = self.backbone(x)
        out = self.decode_head(features)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out


# ----------------- 加载模型权重 -----------------
net = CompleteSegFormer(in_channels=1, num_classes=1).cuda()
# 加载之前训练保存的最佳权重或最终权重
checkpoint = torch.load('SegFormer_Weights/net_final.pth', map_location='cuda')
net.load_state_dict(checkpoint)
net.eval()

# ----------------- 推理预测 -----------------
test_data = TestSet()
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

num_samples = len(data_path)
res = np.zeros((num_samples, 256, 256))

with torch.no_grad():
    for i, data in enumerate(test_loader):
        data = data.cuda()
        pred = net(data)

        pred = pred.squeeze().cpu().numpy()
        pred_binary = np.where(pred < 0.5, 0.0, 1.0)

        res[i] = pred_binary

        plt.imsave(f"{save_pred_dir}/{i + 1}.png", res[i], cmap="gray")
        print(f"Predicted and saved: {save_pred_dir}/{i + 1}.png")