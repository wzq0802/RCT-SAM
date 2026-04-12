import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from build_sam import sam_model_registry
import torch.nn.functional as F
import warnings
from torchvision import transforms
from lora import Linear, MergedLinear
warnings.filterwarnings("ignore")

# 固定随机种子
seed = 82
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# 数据路径
data_path = os.listdir("ori_npy")
data_path.sort(key=lambda x: int(x[:-4]))
data_path1 = os.listdir("seg_npy")
data_path1.sort(key=lambda x: int(x[:-4]))

# 训练验证划分
train_val_ratio = 0.8
train_size = int(train_val_ratio * len(data_path))

train_paths = data_path[:train_size]
val_paths = data_path[train_size:]

# 数据加载器
def default_loader(path):
    data_pil = np.load(f"ori_npy/{path}").reshape((1, 256, 256))
    data_tensor = torch.tensor(data_pil).type(torch.FloatTensor) / 255.0  # 归一化到 [0, 1]

    data_tensor = data_tensor.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
    transform = transforms.Resize((1024, 1024))
    data_tensor = transform(data_tensor)
    return data_tensor

def default_loader1(path):
    data_pil1 = np.load(f"seg_npy/{path}").reshape((1, 256, 256))
    data_tensor1 = torch.tensor(data_pil1).type(torch.FloatTensor)
    return data_tensor1

# 自定义数据集
class NpyDataset(Dataset):
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

def inject_lora_to_sam(sam_model, rank=4):
  
    for name, module in sam_model.image_encoder.named_children():
        if isinstance(module, nn.ModuleList):  # 处理ViT的blocks
            for block in module:
                # 替换QKV投影层为MergedLinear
                block.attn.qkv = MergedLinear(
                    in_features=block.attn.qkv.in_features,
                    out_features=block.attn.qkv.out_features,
                    r=rank,
                    lora_alpha=16,
                    enable_lora=[True, True, True],  # 同时启用Q/K/V的LoRA
                    merge_weights=False
                )
                # 修改MLP层
                block.mlp.lin1 = Linear(
                    in_features=block.mlp.lin1.in_features,
                    out_features=block.mlp.lin1.out_features,
                    r=rank,
                    lora_alpha=16,
                    merge_weights=False
                )
                block.mlp.lin2 = Linear(
                    in_features=block.mlp.lin2.in_features,
                    out_features=block.mlp.lin2.out_features,
                    r=rank,
                    lora_alpha=16,
                    merge_weights=False
                )
    return sam_model


class RCTSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super(RockSAM, self).__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):
        image_embeddings = self.image_encoder(image)  # (B, 256, 64, 64)
        dense_prompt_embeddings = torch.zeros_like(image_embeddings)
        sparse_prompt_embeddings = torch.zeros((image_embeddings.size(0), 0, 256), device=image_embeddings.device)

        masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_prompt_embeddings,  # (B, 0, 256)
            dense_prompt_embeddings=dense_prompt_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        return masks

sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = inject_lora_to_sam(sam, rank=4)

for name, param in sam.named_parameters():
    if 'lora_A' not in name and 'lora_B' not in name:
        param.requires_grad = False
    else:
        print(f"Trainable layer: {name}")

for param in sam.prompt_encoder.parameters():
    param.requires_grad = False

for param in sam.mask_decoder.parameters():
    param.requires_grad = True

rock_sam = RCTSAM(
    image_encoder=sam.image_encoder,
    mask_decoder=sam.mask_decoder,
    prompt_encoder=sam.prompt_encoder,
).to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, rock_sam.parameters()), lr=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5)

# 损失函数
bce_loss = nn.BCEWithLogitsLoss()

def loss(pred, target):
    target = target.squeeze(1)
    loss_bce = bce_loss(pred, target)
    return loss_bce

# 数据加载
train_data = NpyDataset(train_paths)
val_data = NpyDataset(val_paths)

batch_size = 1
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size)

# 训练参数
epochs = 20
best_val_loss = float("inf")

# 保存损失值
Loss_list = []
Val_loss_list = []

# 创建保存目录
os.makedirs("model", exist_ok=True)

for epoch in range(epochs):
    train_loss = 0.0
    val_loss = 0.0
    rock_sam.train()

    for data, label in tqdm(trainloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
        data, label = data.to(device), label.to(device)

        pred = rock_sam(data).squeeze(1)
        loss_value = loss(pred, label)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.item()

    train_loss /= len(trainloader)
    Loss_list.append(train_loss)

    rock_sam.eval()
    with torch.no_grad():
        for data_val, label_val in valloader:
            data_val, label_val = data_val.to(device), label_val.to(device)
            pred_val = rock_sam(data_val).squeeze(1)
            val_loss += loss(pred_val, label_val).item()

    val_loss /= len(valloader)
    Val_loss_list.append(val_loss)

    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(rock_sam.state_dict(), "model/best_model.pth")
        print(f"Best model saved with val loss: {val_loss:.6f}")


    scheduler.step(val_loss)

# 保存最终模型
torch.save(rock_sam.state_dict(), "model/sam/final_model.pth")
print("Final model saved")


print("Training completed!")