import sys
import time
import warnings
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch
import torch.nn.functional as F
from DMTNN import ImprovedLinkNet
import torch.nn as nn
warnings.filterwarnings('ignore')
seed = 82
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
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
    # data_tensor = torch.tensor(data_pil).type(torch.FloatTensor)/255

    data_min = np.min(data_pil)
    data_max = np.max(data_pil)
    if data_max > data_min:
        data_normalized = (data_pil - data_min) / (data_max - data_min)
    else:
        data_normalized = np.zeros_like(data_pil)

    # 转换为 Tensor
    data_tensor = torch.tensor(data_pil, dtype=torch.float32)
    data_standardized = data_tensor
    # 转换为 Tensor 并返回
    data_tensor = torch.tensor(data_standardized, dtype=torch.float32)

    return data_tensor

def default_loader1(path):
    data_pil1 = np.load("data/50Berea/seg_npy/%s" % (path)).reshape((1, 256, 256))
    data_tensor1 = torch.tensor(data_pil1).type(torch.FloatTensor)
    return data_tensor1


class trainset(Dataset):
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



train_data = trainset(train_paths)
val_data = trainset(val_paths)


batch_size = 4

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size)

net=ImprovedLinkNet().cuda()

optimizer = optim.Adam(net.parameters(), lr=0.0005, betas=(0.5, 0.999))
# optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999),weight_decay=1e-5)   ##Regularization to prevent overfitting on small datasets

mse = nn.MSELoss()


epochs = 10
Loss_list = []
Val_loss_list = []
best_val_loss = float('inf')
best_epoch = 0


# 训练循环
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    best_val_loss_in_100_epochs = float('inf')  # 初始化100轮中的最佳验证损失
    best_model_in_100_epochs = None  # 用于存储当前100轮中的最佳模型状态

    net.train()
    for i, (data, label) in enumerate(trainloader):
        data = data.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        pred = net(data)
        # loss =  F.binary_cross_entropy_with_logits(pred, label)
        loss = mse(pred, label)



        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        sys.stdout.write(
            "[Epoch %d/%d] [Batch: %d/%d] [loss: %f]\n" %(
                epoch, epochs, len(trainloader), i ,loss.item(),
            )
        )

    # 记录训练损失
    Loss_list.append(train_loss / len(trainloader))

    # 进行验证
    net.eval()
    with torch.no_grad():
        for data_val, label_val in valloader:
            data_val = data_val.cuda()
            label_val = label_val.cuda()
            pred_val = net(data_val)
            # val_loss +=  F.binary_cross_entropy_with_logits(pred_val, label_val).item()
            val_loss +=mse(pred_val, label_val).item()

    val_loss /= len(valloader)
    Val_loss_list.append(val_loss)

    # 判断是否为当前100轮中的最佳模型
    if val_loss < best_val_loss_in_100_epochs:
        best_val_loss_in_100_epochs = val_loss
        best_model_in_100_epochs = net.state_dict()

    # 每100轮时保存当前100轮中的最佳模型
    if (epoch + 1) % 20 == 0:
        torch.save(best_model_in_100_epochs, f"DMTNN/best_net_{epoch + 1}_epochs.pth")
        print(f"Saved the best model of last 100 epochs at epoch {epoch + 1}")




# 保存最终模型
torch.save(net.state_dict(), "DMTNN/net_final.pth")

# 保存损失记录
np.savetxt("DMTNN/Train_Loss.csv", np.array(Loss_list))
np.savetxt("DMTNN/Val_Loss.csv", np.array(Val_loss_list))

