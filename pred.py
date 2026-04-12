import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from build_sam import sam_model_registry
from lora import Linear, MergedLinear
from skimage import io, transform


def load_model_with_lora(model_type, sam_checkpoint, lora_path):
    """加载SAM基础模型并注入LoRA结构与权重"""
    sam = sam_model_registry[model_type](checkpoint=None).to(device)

    # 替换 image_encoder 中的 QKV 和 MLP 层
    for name, module in sam.image_encoder.named_children():
        if isinstance(module, nn.ModuleList):
            for block in module:
                block.attn.qkv = MergedLinear(
                    in_features=block.attn.qkv.in_features,
                    out_features=block.attn.qkv.out_features,
                    r=4, lora_alpha=16,
                    enable_lora=[True, True, True],
                    merge_weights=False
                )
                block.mlp.lin1 = Linear(
                    in_features=block.mlp.lin1.in_features,
                    out_features=block.mlp.lin1.out_features,
                    r=4, lora_alpha=16, merge_weights=False
                )
                block.mlp.lin2 = Linear(
                    in_features=block.mlp.lin2.in_features,
                    out_features=block.mlp.lin2.out_features,
                    r=4, lora_alpha=16, merge_weights=False
                )

    # 加载主模型权重
    sam.load_state_dict(torch.load(sam_checkpoint, map_location=device))

    # 加载LoRA权重
    lora_weights = torch.load(lora_path, map_location=device)
    for name, param in sam.image_encoder.named_parameters():
        if name in lora_weights:
            param.data.copy_(lora_weights[name])

    return sam


def preprocess_image(image_path):
    img_np = np.array(Image.open(image_path))
    if img_np.ndim == 2:
        img_np = np.repeat(img_np[:, :, None], 3, axis=-1)
    elif img_np.shape[-1] == 4:
        img_np = img_np[..., :3]

    H, W, _ = img_np.shape
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1024, 1024), antialias=True)
    ])
    img_tensor = transform(img_np).unsqueeze(0).to(device)
    return img_tensor, (H, W), img_np


class RCTSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super(RCTSAM, self).__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # 冻结 image_encoder 和 prompt_encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):
        # 提取图像嵌入
        with torch.no_grad():
            image_embeddings = self.image_encoder(image)

        dense_prompt_embeddings = torch.zeros_like(image_embeddings)
        sparse_prompt_embeddings = torch.zeros((image_embeddings.size(0), 0, 256),
                                  device=image_embeddings.device)

        # 调用 mask_decoder
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
        )

        high_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        return high_res_masks


def rocksam_inference(rocksam_model, img_tensor, H, W):

    high_res_masks = rocksam_model(img_tensor)
    binary_segmentation = (
        F.interpolate(high_res_masks, size=(H, W), mode="bilinear", align_corners=False)
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    binary_segmentation = (binary_segmentation > 0.5).astype(np.uint8)
    return binary_segmentation


def calculate_metrics(pred_mask, true_mask):
    """计算 IoU、Dice 和 Accuracy"""
    pred_mask = pred_mask.astype(np.uint8)
    true_mask = true_mask.astype(np.uint8)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else 0

    dice = (2.0 * intersection) / (pred_mask.sum() + true_mask.sum()) if (pred_mask.sum() + true_mask.sum()) != 0 else 0

    accuracy = np.sum(pred_mask == true_mask) / (true_mask.shape[0] * true_mask.shape[1])

    return iou, dice, accuracy


if __name__ == "__main__":
    # 设置路径
    input_folder = "ori_png"  # 输入图像文件夹
    true_mask_folder = "true_png"  # 真实掩码文件夹
    output_folder = "pred"  # 预测结果保存文件夹
    sam_checkpoint = "point/SAM_lora/sam_epoch_50.pth"  # SAM模型权重
    lora_weights = "lora_epoch_50.pth"  # LoRA权重路径

    os.makedirs(output_folder, exist_ok=True)

    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 初始化模型
    sam = load_model_with_lora("vit_b", sam_checkpoint, lora_weights)
    rocksam_model = RockSAM(
        image_encoder=sam.image_encoder,
        mask_decoder=sam.mask_decoder,
        prompt_encoder=sam.prompt_encoder,
    ).to(device)
    rocksam_model.eval()

    # 计算总指标
    total_iou, total_dice, total_acc, count = 0, 0, 0, 0

    for file_name in sorted(os.listdir(input_folder)):
        if not file_name.endswith(".png"):  # 跳过非PNG文件
            continue

        input_image_path = os.path.join(input_folder, file_name)
        true_mask_path = os.path.join(true_mask_folder, file_name.replace(".png", ".npy"))  # 假设掩码是.npy格式
        output_image_path = os.path.join(output_folder, file_name)

        # 预处理图像
        img_tensor, original_size, _ = preprocess_image(input_image_path)

        # 推理
        binary_mask = rocksam_inference(rocksam_model, img_tensor, original_size[0], original_size[1])

        # 加载真实掩码
        true_mask = np.load(true_mask_path).astype(np.uint8)
        if true_mask.shape != binary_mask.shape:
            true_mask = Image.fromarray(true_mask).resize(binary_mask.shape[::-1], Image.NEAREST)
            true_mask = np.array(true_mask)

        # 计算评估指标
        iou, dice, acc = calculate_metrics(binary_mask, true_mask)
        total_iou += iou
        total_dice += dice
        total_acc += acc
        count += 1

        # 保存预测结果
        Image.fromarray(binary_mask * 255).save(output_image_path)

        print(f"Processed {file_name}: IoU={iou:.4f}, Dice={dice:.4f}, Acc={acc:.4f}")

    # 打印平均指标
    avg_iou = total_iou / count if count > 0 else 0
    avg_dice = total_dice / count if count > 0 else 0
    avg_acc = total_acc / count if count > 0 else 0
    print(f"\nAverage Metrics:")
    print(f"IoU: {avg_iou:.4f}")
    print(f"Dice: {avg_dice:.4f}")
    print(f"Accuracy: {avg_acc:.4f}")