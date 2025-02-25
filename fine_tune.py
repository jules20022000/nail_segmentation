import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

#### NOTE : FINE TUNING of the SAM2 model with the nail dataset ####

data_dir = r"./data"
data = []
for file_name in os.listdir(os.path.join(data_dir, "images")):
    data.append(
        {
            "image": os.path.join(data_dir, "images", file_name),
            "annotation": os.path.join(data_dir, "mask_images", file_name),
        }
    )

# Split the data into 80% training and 20% validation.
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

def read_sample(entry):
    """
    Read and process a single sample given an entry dictionary.
    """
    img = cv2.imread(entry["image"])[..., ::-1]  # convert BGR to RGB
    ann_map = cv2.imread(entry["annotation"], cv2.IMREAD_GRAYSCALE)  # annotation map

    # Resize image and annotation to have a max dimension of 1024.
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    ann_map = cv2.resize(
        ann_map,
        (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
        interpolation=cv2.INTER_NEAREST,
    )

    binary_mask = (ann_map > 100).astype(np.uint8)
    masks = [binary_mask]  # list of masks in the image
    points = []

    # Get all coordinates where the mask is present
    coords = np.argwhere(binary_mask > 0)
    if len(coords) > 0:
        yx = coords[np.random.randint(len(coords))]
        points.append([yx[1], yx[0]])  # store as (x, y)
    labels = np.ones(len(points), dtype=int)
    return img, np.array(masks), np.array(points), labels

def read_batch(data_list):
    """Choose a random entry from a list and process it."""
    entry = data_list[np.random.randint(len(data_list))]
    return read_sample(entry)

def evaluate_model(predictor, val_set):
    """
    Evaluate the model on the validation set.
    Returns the average loss and average IOU over all validation samples.
    """
    predictor.model.eval()  # set evaluation mode
    val_losses = []
    val_ious = []
    with torch.no_grad():
        for entry in tqdm(val_set):
            image, mask, input_point, input_label = read_sample(entry)
            predictor.set_image(image)
            
            with torch.amp.autocast("cuda"):
                # Prepare prompts
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                    input_point, input_label, box=None, mask_logits=None, normalize_coords=True
                )
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels), boxes=None, masks=None
                )
                batched_mode = unnorm_coords.shape[0] > 1  # multi-object case
                high_res_features = [
                    feat_level[-1].unsqueeze(0)
                    for feat_level in predictor._features["high_res_feats"]
                ]
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )
                prd_masks = predictor._transforms.postprocess_masks(
                    low_res_masks, predictor._orig_hw[-1]
                )
                gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                prd_mask = torch.sigmoid(prd_masks[:, 0])  # probability map
                seg_loss = (
                    -gt_mask * torch.log(prd_mask + 1e-5)
                    - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)
                ).mean()
                inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                iou = inter / (
                    gt_mask.sum(1).sum(1)
                    + (prd_mask > 0.5).sum(1).sum(1)
                    - inter
                )
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                loss_val = seg_loss + score_loss * 0.05
                val_losses.append(loss_val.item())
                val_ious.append(iou.mean().item())
                
    predictor.model.train()  # revert back to training mode
    return np.mean(val_losses), np.mean(val_ious)

def visualize(img, masks, points):
    """
    Visualize the original image, the mask, and the mask with points.
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    if len(masks) > 0:
        plt.imshow(masks[0] * 255, cmap="gray")
    plt.title("Mask Only")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    if len(masks) > 0:
        plt.imshow(masks[0] * 255, cmap="gray")
    for point in points:
        if len(point) > 1:
            plt.plot(point[0], point[1], "ro")
    plt.title("Mask with Points")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Example visualization (optional)
# img, masks, points, _ = read_sample(train_data[0])
# visualize(img, masks, points)

# Load the SAM2 model
checkpoint = "./sam2.1_hiera_base_plus.pt"
model_cfg = "/configs/sam2.1/sam2.1_hiera_b+.yaml"
sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Enable training for the mask decoder and prompt encoder
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)

optimizer = torch.optim.AdamW(
    params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5
)
scaler = torch.amp.GradScaler()  # mixed precision

# Training loop variables for loss tracking
running_train_loss = 0.0
count_train = 0

# Main training loop
num_iterations = 10_000
eval_interval = 2000 # evaluate every 1000 iterations

mean_iou = 0.0  # running mean of IOU (for logging)

for itr in tqdm(range(num_iterations)):
    with torch.amp.autocast("cuda"):
        # Sample a training batch
        image, mask, input_point, input_label = read_batch(train_data)
        if mask.shape[0] == 0:
            continue  # skip empty masks
        predictor.set_image(image)

        # Prepare prompts
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point, input_label, box=None, mask_logits=None, normalize_coords=True
        )
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None
        )
        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [
            feat_level[-1].unsqueeze(0)
            for feat_level in predictor._features["high_res_feats"]
        ]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        prd_masks = predictor._transforms.postprocess_masks(
            low_res_masks, predictor._orig_hw[-1]
        )

        # Compute segmentation loss (binary cross entropy style)
        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        seg_loss = (
            -gt_mask * torch.log(prd_mask + 1e-5)
            - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)
        ).mean()

        # Compute IOU and score loss
        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (
            gt_mask.sum(1).sum(1)
            + (prd_mask > 0.5).sum(1).sum(1)
            - inter
        )
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.05

    running_train_loss += loss.item()
    count_train += 1

    # Backpropagation and optimization step
    predictor.model.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Update a running mean of IOU (for logging)
    mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

    # Periodically evaluate and print metrics
    if itr % eval_interval == 0 and itr > 0:
        # Save checkpoint
        torch.save(predictor.model.state_dict(), "model.torch")
        # Compute average training loss for the interval
        avg_train_loss = running_train_loss / count_train if count_train > 0 else float('nan')
        
        # Run evaluation on the entire validation set
        val_loss, val_iou = evaluate_model(predictor, val_data)
        
        print(
            f"Step {itr}: Train Loss = {avg_train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}, Val IOU = {val_iou:.4f}"
        )
        
        # Reset running training loss counters for the next interval
        running_train_loss = 0.0
        count_train = 0

        # Optionally, you can also print the running mean IOU from training
        print(f"Running Train IOU = {mean_iou:.4f}")
        
    
# Save the final model
torch.save(predictor.model.state_dict(), "model.torch")
