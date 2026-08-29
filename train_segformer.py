import sys
import time
import warnings
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 从同目录下的文件中导入 SegFormer 架构组件
from mix_transformer import mit_b0
from segformer_head import SegFormerHead

warnings.filterwarnings('ignore')
seed = 82
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# 创建权重与日志输出目录
save_dir = "SegFormer_Weights"
os.makedirs(save_dir, exist_ok=True)

data_path = os.listdir("data/50Berea/ori_npy")
data_path.sort(key=lambda x: int(x[:-4]))
data_path1 = os.listdir("data/50Berea/seg_npy")
data_path1.sort(key=lambda x: int(x[:-4]))

train_val_ratio = 0.8
train_size = int(train_val_ratio * len(data_path))

train_paths = data_path[:train_size]
val_paths = data_path[train_size:]


def default_loader(path):
    data_pil = np.load("data/50Berea/ori_npy/%s" % (path)).reshape((1, 256, 256))
    data_tensor = torch.tensor(data_pil, dtype=torch.float32)
    return data_tensor


def default_loader1(path):
    data_pil1 = np.load("data/50Berea/seg_npy/%s" % (path)).reshape((1, 256, 256))
    data_tensor1 = torch.tensor(data_pil1, dtype=torch.float32)
    return data_tensor1


class TrainSet(Dataset):
    def __init__(self, paths, loader=default_loader, loader1=default_loader1):
        self.images = paths
        self.loader = loader
        self.loader1 = loader1

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.loader1(fn)
        return img, target

    def __len__(self):
        return len(self.images)


train_data = TrainSet(train_paths)
val_data = TrainSet(val_paths)

batch_size = 4
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size)


# ----------------- 构建完整的 SegFormer 模型 -----------------
class CompleteSegFormer(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        # 1. 实例化 Backbone (将 in_chans 设为 1 以适配单通道灰度数据)
        self.backbone = mit_b0(in_chans=in_channels)

        # 2. 实例化 Decode Head (对应 mit_b0 的多尺度输出通道配置)
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
        # 获取 4 个 stage 的特征图 outs: [C1, C2, C3, C4]
        features = self.backbone(x)
        # 解码融合 (输出尺寸为原图的 1/4)
        out = self.decode_head(features)
        # 上采样回原图分辨率 (256, 256)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out


net = CompleteSegFormer(in_channels=1, num_classes=1).cuda()

# SegFormer 推荐使用 AdamW 优化器
optimizer = optim.AdamW(net.parameters(), lr=0.0005, weight_decay=0.01)
mse = nn.BCEWithLogitsLoss()

epochs = 20
Loss_list = []
Val_loss_list = []
best_val_loss = float('inf')

# ----------------- 训练与验证循环 -----------------
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0

    net.train()
    for i, (data, label) in enumerate(trainloader):
        data = data.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        pred = net(data)
        loss = mse(pred, label)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        sys.stdout.write(
            "[Epoch %d/%d] [Batch: %d/%d] [loss: %f]\n" % (
                epoch + 1, epochs, len(trainloader), i + 1, loss.item()
            )
        )

    # 记录训练损失
    avg_train_loss = train_loss / len(trainloader)
    Loss_list.append(avg_train_loss)

    # 验证
    net.eval()
    with torch.no_grad():
        for data_val, label_val in valloader:
            data_val = data_val.cuda()
            label_val = label_val.cuda()
            pred_val = net(data_val)
            val_loss += mse(pred_val, label_val).item()

    avg_val_loss = val_loss / len(valloader)
    Val_loss_list.append(avg_val_loss)
    print(f"--> Epoch {epoch + 1} Done | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(net.state_dict(), f"{save_dir}/best_net.pth")
        print(f"--- Saved New Best Model (Val Loss: {best_val_loss:.6f}) ---")

# 保存最终模型及损失
torch.save(net.state_dict(), f"{save_dir}/net_final.pth")
np.savetxt(f"{save_dir}/Train_Loss.csv", np.array(Loss_list))
np.savetxt(f"{save_dir}/Val_Loss.csv", np.array(Val_loss_list))