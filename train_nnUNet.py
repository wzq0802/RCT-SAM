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

from generic_UNet import Generic_UNet

warnings.filterwarnings('ignore')
seed = 82
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


save_dir = "nnUNet_Weights"
os.makedirs(save_dir, exist_ok=True)

data_path = os.listdir("ori_npy")
data_path.sort(key=lambda x: int(x[:-4]))
data_path1 = os.listdir("seg_npy")
data_path1.sort(key=lambda x: int(x[:-4]))

train_val_ratio = 0.8
train_size = int(train_val_ratio * len(data_path))

train_paths = data_path[:train_size]
val_paths = data_path[train_size:]

def default_loader(path):
    data_pil = np.load("ori_npy/%s" % (path)).reshape((1, 256, 256))
    data_tensor = torch.tensor(data_pil, dtype=torch.float32)
    return data_tensor

def default_loader1(path):
    data_pil1 = np.load("seg_npy/%s" % (path)).reshape((1, 256, 256))
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

net = Generic_UNet(
    input_channels=1,
    base_num_features=32,
    num_classes=1,
    num_pool=5,
    num_conv_per_stage=2,
    feat_map_mul_on_downscale=2,
    conv_op=nn.Conv2d,
    norm_op=nn.BatchNorm2d,
    dropout_op=nn.Dropout2d,
    nonlin=nn.LeakyReLU,
    deep_supervision=False,
    final_nonlin=lambda x: x 
).cuda()

optimizer = optim.AdamW(net.parameters(), lr=0.0005, weight_decay=1e-4)
mse = nn.BCEWithLogitsLoss()

epochs = 20
Loss_list = []
Val_loss_list = []
best_val_loss = float('inf')

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

    avg_train_loss = train_loss / len(trainloader)
    Loss_list.append(avg_train_loss)

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

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(net.state_dict(), f"{save_dir}/best_net.pth")
        print(f"--- Saved New Best Model (Val Loss: {best_val_loss:.6f}) ---")

torch.save(net.state_dict(), f"{save_dir}/net_final.pth")
np.savetxt(f"{save_dir}/Train_Loss.csv", np.array(Loss_list))
np.savetxt(f"{save_dir}/Val_Loss.csv", np.array(Val_loss_list))
