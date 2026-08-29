import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image
from build_sam import sam_model_registry
from lora import Linear, MergedLinear


# ============================================================
# 1. Global configuration
# ============================================================

SEED = 82

SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
MODEL_WEIGHT = "model/epoch_20.pth"

TEST_ROOT = "test"
OUTPUT_ROOT = "test_pred"

LORA_RANK = 4
LORA_ALPHA = 16

# Keep the original logit threshold
PREDICTION_THRESHOLD = 0.5

# SAM input size
INPUT_SIZE = (1024, 1024)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. Reproducibility
# ============================================================

def set_random_seed(seed):
    """Fix random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. LoRA injection
# ============================================================

def inject_lora_to_sam(
    sam_model,
    rank=4,
    lora_alpha=16
):
    """
    Inject LoRA modules into the QKV projections and MLP layers
    of the SAM image encoder.

    The architecture must be identical to that used during training.
    """

    for _, module in sam_model.image_encoder.named_children():

        if isinstance(module, nn.ModuleList):

            for block in module:

                block.attn.qkv = MergedLinear(
                    in_features=block.attn.qkv.in_features,
                    out_features=block.attn.qkv.out_features,
                    r=rank,
                    lora_alpha=lora_alpha,
                    enable_lora=[True, True, True],
                    merge_weights=False
                )

                block.mlp.lin1 = Linear(
                    in_features=block.mlp.lin1.in_features,
                    out_features=block.mlp.lin1.out_features,
                    r=rank,
                    lora_alpha=lora_alpha,
                    merge_weights=False
                )

                block.mlp.lin2 = Linear(
                    in_features=block.mlp.lin2.in_features,
                    out_features=block.mlp.lin2.out_features,
                    r=rank,
                    lora_alpha=lora_alpha,
                    merge_weights=False
                )

    return sam_model


# ============================================================
# 4. RockSAM model
# ============================================================

class RockSAM(nn.Module):
    """
    Prompt-free RockSAM using fixed zero prompt embeddings.
    """

    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image):

        image_embeddings = self.image_encoder(image)

        batch_size = image_embeddings.size(0)
        embedding_channels = image_embeddings.size(1)

        # Fixed zero dense-prompt embeddings
        dense_prompt_embeddings = torch.zeros_like(
            image_embeddings
        )

        # Fixed empty sparse-prompt embeddings
        sparse_prompt_embeddings = torch.zeros(
            (
                batch_size,
                0,
                embedding_channels
            ),
            dtype=image_embeddings.dtype,
            device=image_embeddings.device
        )

        masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False
        )

        return masks


# ============================================================
# 5. Image preprocessing
# ============================================================

def preprocess_image(image_path):
    """
    Load and preprocess one grayscale CT image.

    The preprocessing is identical to the original prediction code:
    1. Convert the image to grayscale.
    2. Normalize grayscale values to [0, 1].
    3. Repeat the grayscale channel three times.
    4. Resize the image to 1024 x 1024.
    """

    image = Image.open(image_path).convert("L")

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image_tensor = torch.from_numpy(
        image_array
    ).unsqueeze(0)

    # Convert grayscale image to three channels
    image_tensor = image_tensor.repeat(
        3,
        1,
        1
    )

    resize_transform = transforms.Resize(
        INPUT_SIZE
    )

    image_tensor = resize_transform(
        image_tensor
    )

    # Add the batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


# ============================================================
# 6. Reference-mask loading
# ============================================================

def load_ground_truth(gt_path):
    """
    Load the binary reference mask using the original encoding rule.
    """

    gt_image = Image.open(gt_path).convert("L")

    gt_array = np.asarray(
        gt_image
    )

    # Keep exactly the same processing as the original code
    gt_binary = (
        gt_array > 0
    ).astype(np.float32)

    gt_tensor = torch.from_numpy(
        gt_binary
    )

    return gt_tensor


# ============================================================
# 7. Model prediction
# ============================================================

def predict_with_model(
    model,
    image_tensor
):
    """
    Generate a binary prediction using the original decision rule.

    The raw output logits are directly compared with 0.5.
    No sigmoid operation is applied.
    """

    model.eval()

    with torch.no_grad():

        output = model(
            image_tensor
        )

        output = output.squeeze().cpu()

        # Keep exactly the same threshold as the original code
        prediction = (
            output > PREDICTION_THRESHOLD
        ).float()

    return prediction


# ============================================================
# 8. Metric calculation
# ============================================================

def calculate_metrics(
    prediction,
    target
):
    """
    Calculate IoU, Dice, precision, and recall.
    """

    prediction = (
        prediction > 0.5
    ).bool()

    target = (
        target > 0.5
    ).bool()

    tp = torch.logical_and(
        prediction,
        target
    ).sum().item()

    fp = torch.logical_and(
        prediction,
        ~target
    ).sum().item()

    fn = torch.logical_and(
        ~prediction,
        target
    ).sum().item()

    eps = 1e-8

    iou = tp / (
        tp + fp + fn + eps
    )

    dice = 2.0 * tp / (
        2.0 * tp + fp + fn + eps
    )

    precision = tp / (
        tp + fp + eps
    )

    recall = tp / (
        tp + fn + eps
    )

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall)
    }


# ============================================================
# 9. Model initialization
# ============================================================

def build_model():
    """
    Initialize SAM, inject LoRA, and load the trained weights.
    """

    if not os.path.exists(SAM_CHECKPOINT):
        raise FileNotFoundError(
            f"SAM checkpoint does not exist: {SAM_CHECKPOINT}"
        )

    if not os.path.exists(MODEL_WEIGHT):
        raise FileNotFoundError(
            f"Model weight does not exist: {MODEL_WEIGHT}"
        )

    print("=" * 70)
    print("Initializing model")
    print("=" * 70)

    base_sam = sam_model_registry[
        MODEL_TYPE
    ](
        checkpoint=SAM_CHECKPOINT
    )

    base_sam = inject_lora_to_sam(
        sam_model=base_sam,
        rank=LORA_RANK,
        lora_alpha=LORA_ALPHA
    )

    rock_sam = RockSAM(
        image_encoder=base_sam.image_encoder,
        mask_decoder=base_sam.mask_decoder,
        prompt_encoder=base_sam.prompt_encoder
    ).to(DEVICE)

    print(f"Loading model weight: {MODEL_WEIGHT}")

    checkpoint = torch.load(
        MODEL_WEIGHT,
        map_location=DEVICE
    )

    # Support a checkpoint that contains a "state_dict" key
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        checkpoint = checkpoint["state_dict"]

    rock_sam.load_state_dict(
        checkpoint,
        strict=True
    )

    rock_sam.eval()

    print("Model weight loaded successfully.")
    print(f"Device: {DEVICE}")

    return rock_sam


# ============================================================
# 10. Dataset evaluation
# ============================================================

def evaluate_dataset(
    model,
    dataset_name,
    original_folder,
    ground_truth_folder,
    output_folder
):
    """
    Evaluate one dataset and return image-level results.
    """

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    if not os.path.isdir(original_folder):
        print(
            f"Warning: original-image folder does not exist: "
            f"{original_folder}"
        )
        return []

    if not os.path.isdir(ground_truth_folder):
        print(
            f"Warning: reference-mask folder does not exist: "
            f"{ground_truth_folder}"
        )
        return []

    image_files = sorted([
        filename
        for filename in os.listdir(original_folder)
        if filename.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff"
            )
        )
    ])

    if len(image_files) == 0:
        print(
            f"Warning: no test image was found in "
            f"{original_folder}"
        )
        return []

    print("\n" + "=" * 80)
    print(f"Evaluating dataset: {dataset_name}")
    print(f"Number of candidate images: {len(image_files)}")
    print("=" * 80)

    image_results = []

    for index, image_name in enumerate(image_files):

        image_path = os.path.join(
            original_folder,
            image_name
        )

        gt_path = os.path.join(
            ground_truth_folder,
            image_name
        )

        if not os.path.exists(gt_path):
            print(
                f"Warning: corresponding reference mask "
                f"was not found: {gt_path}"
            )
            continue

        # Load input image
        image_tensor = preprocess_image(
            image_path
        ).to(DEVICE)

        # Load reference mask
        gt_mask = load_ground_truth(
            gt_path
        )

        # Generate prediction
        prediction = predict_with_model(
            model=model,
            image_tensor=image_tensor
        )

        # The original evaluation assumes that the output mask
        # and reference mask have the same spatial dimensions
        if prediction.shape != gt_mask.shape:
            raise ValueError(
                f"Prediction and reference-mask sizes do not match. "
                f"Dataset: {dataset_name}, "
                f"Image: {image_name}, "
                f"Prediction size: {tuple(prediction.shape)}, "
                f"Reference size: {tuple(gt_mask.shape)}"
            )

        # Calculate four metrics
        metrics = calculate_metrics(
            prediction=prediction,
            target=gt_mask
        )

        result_row = {
            "Dataset": dataset_name,
            "Image": image_name,
            "IoU": metrics["IoU"],
            "Dice": metrics["Dice"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"]
        }

        image_results.append(
            result_row
        )

        # Save prediction mask
        prediction_image = (
            prediction.numpy() * 255
        ).astype(np.uint8)

        save_path = os.path.join(
            output_folder,
            os.path.splitext(image_name)[0] + ".png"
        )

        Image.fromarray(
            prediction_image
        ).save(save_path)

        print(
            f"[{index + 1:4d}/{len(image_files)}] "
            f"{image_name} | "
            f"IoU={metrics['IoU']:.4f} | "
            f"Dice={metrics['Dice']:.4f} | "
            f"Precision={metrics['Precision']:.4f} | "
            f"Recall={metrics['Recall']:.4f}"
        )

    return image_results


# ============================================================
# 11. Generate dataset summary
# ============================================================

def generate_dataset_summary(
    image_results
):
    """
    Calculate the mean IoU, Dice, precision, and recall
    for each dataset.
    """

    if len(image_results) == 0:
        return pd.DataFrame()

    image_dataframe = pd.DataFrame(
        image_results
    )

    summary_dataframe = (
        image_dataframe
        .groupby(
            "Dataset",
            sort=False
        )
        .agg(
            N=("Image", "count"),
            IoU=("IoU", "mean"),
            Dice=("Dice", "mean"),
            Precision=("Precision", "mean"),
            Recall=("Recall", "mean")
        )
        .reset_index()
    )

    return summary_dataframe


# ============================================================
# 12. Save results to Excel
# ============================================================

def save_results_to_excel(
    image_results,
    output_excel
):
    """
    Save dataset-level and image-level results to one Excel file.
    """

    if len(image_results) == 0:
        print("No valid result was available for Excel output.")
        return

    image_dataframe = pd.DataFrame(
        image_results
    )

    summary_dataframe = generate_dataset_summary(
        image_results
    )

    # Round values to six decimal places in the Excel file
    metric_columns = [
        "IoU",
        "Dice",
        "Precision",
        "Recall"
    ]

    image_dataframe[
        metric_columns
    ] = image_dataframe[
        metric_columns
    ].round(6)

    summary_dataframe[
        metric_columns
    ] = summary_dataframe[
        metric_columns
    ].round(6)

    with pd.ExcelWriter(
        output_excel,
        engine="openpyxl"
    ) as writer:

        summary_dataframe.to_excel(
            writer,
            sheet_name="Dataset Summary",
            index=False
        )

        image_dataframe.to_excel(
            writer,
            sheet_name="Image Details",
            index=False
        )

        # Adjust Excel column widths
        summary_sheet = writer.sheets[
            "Dataset Summary"
        ]

        summary_sheet.column_dimensions["A"].width = 25
        summary_sheet.column_dimensions["B"].width = 10
        summary_sheet.column_dimensions["C"].width = 15
        summary_sheet.column_dimensions["D"].width = 15
        summary_sheet.column_dimensions["E"].width = 15
        summary_sheet.column_dimensions["F"].width = 15

        detail_sheet = writer.sheets[
            "Image Details"
        ]

        detail_sheet.column_dimensions["A"].width = 25
        detail_sheet.column_dimensions["B"].width = 18
        detail_sheet.column_dimensions["C"].width = 15
        detail_sheet.column_dimensions["D"].width = 15
        detail_sheet.column_dimensions["E"].width = 15
        detail_sheet.column_dimensions["F"].width = 15

    print(f"Excel results saved to: {output_excel}")


# ============================================================
# 13. Print final summary
# ============================================================

def print_final_summary(
    summary_dataframe
):
    """
    Print dataset-level mean metrics.
    """

    if summary_dataframe.empty:
        return

    print("\n")
    print("=" * 100)
    print("Dataset-level segmentation results")
    print("=" * 100)

    print(
        f"{'Dataset':<25}"
        f"{'N':>8}"
        f"{'IoU':>15}"
        f"{'Dice':>15}"
        f"{'Precision':>15}"
        f"{'Recall':>15}"
    )

    print("-" * 100)

    for _, row in summary_dataframe.iterrows():

        print(
            f"{row['Dataset']:<25}"
            f"{int(row['N']):>8}"
            f"{row['IoU']:>15.4f}"
            f"{row['Dice']:>15.4f}"
            f"{row['Precision']:>15.4f}"
            f"{row['Recall']:>15.4f}"
        )


# ============================================================
# 14. Main function
# ============================================================

def main():

    set_random_seed(
        SEED
    )

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    if not os.path.isdir(TEST_ROOT):
        raise FileNotFoundError(
            f"Test root does not exist: {TEST_ROOT}"
        )

    # Initialize the trained model
    rock_sam = build_model()

    # Identify all test dataset folders automatically
    dataset_names = sorted([
        folder_name
        for folder_name in os.listdir(TEST_ROOT)
        if os.path.isdir(
            os.path.join(
                TEST_ROOT,
                folder_name
            )
        )
    ])

    if len(dataset_names) == 0:
        raise RuntimeError(
            f"No dataset folder was found in {TEST_ROOT}"
        )

    print("\n" + "=" * 70)
    print(f"Number of test datasets: {len(dataset_names)}")
    print("=" * 70)

    for dataset_name in dataset_names:
        print(f"  - {dataset_name}")

    all_image_results = []

    # Evaluate every dataset
    for dataset_name in dataset_names:

        dataset_root = os.path.join(
            TEST_ROOT,
            dataset_name
        )

        original_folder = os.path.join(
            dataset_root,
            "ori_png"
        )

        ground_truth_folder = os.path.join(
            dataset_root,
            "seg_png"
        )

        output_folder = os.path.join(
            OUTPUT_ROOT,
            dataset_name
        )

        dataset_image_results = evaluate_dataset(
            model=rock_sam,
            dataset_name=dataset_name,
            original_folder=original_folder,
            ground_truth_folder=ground_truth_folder,
            output_folder=output_folder
        )

        all_image_results.extend(
            dataset_image_results
        )

    if len(all_image_results) == 0:
        print("No valid test result was obtained.")
        return

    # Generate and print summary
    summary_dataframe = generate_dataset_summary(
        all_image_results
    )

    print_final_summary(
        summary_dataframe
    )

    # Save all results into one Excel file
    output_excel = os.path.join(
        OUTPUT_ROOT,
        "segmentation_metrics.xlsx"
    )

    save_results_to_excel(
        image_results=all_image_results,
        output_excel=output_excel
    )

    print("\n" + "=" * 70)
    print("Evaluation completed.")
    print(f"Prediction masks: {OUTPUT_ROOT}")
    print(f"Excel file: {output_excel}")
    print("=" * 70)


if __name__ == "__main__":
    main()
