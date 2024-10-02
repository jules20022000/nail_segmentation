import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

#### NOTE : FINE TUNING of the SAM2 model with the nail dataset ####

data_dir = r"C:\Users\jules\OneDrive\Documents\EPFL\these\master_these\nail_detect\data"
data = []
for file_name in os.listdir(os.path.join(data_dir, "images")):
    data.append(
        {
            "image": os.path.join(data_dir, "images", file_name),
            "annotation": os.path.join(data_dir, "mask_images", file_name),
        }
    )


def read_batch(data):

    #  select image and its annotation
    ent = data[np.random.randint(len(data))]  # choose random entry
    img = cv2.imread(ent["image"])[..., ::-1]  # read image (BGR to RGB)
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)  # read annotation map

    # resize image and annotation
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])  # scaling factor
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    ann_map = cv2.resize(
        ann_map,
        (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),
        interpolation=cv2.INTER_NEAREST,
    )

    binary_mask = (ann_map > 100).astype(np.uint8)
    # Process the annotation map to identify the mask
    masks = [binary_mask]  # List of masks present in the image
    points = []

    # Get all coordinates within the binary mask
    coords = np.argwhere(binary_mask > 0)  # Coordinates where mask is present

    # Choose a random point within the binary mask
    if len(coords) > 0:  # Make sure there are coordinates in the mask
        yx = np.array(
            coords[np.random.randint(len(coords))]
        )  # Choose a random point (y, x)
        points.append([yx[1], yx[0]])  # Store the point as (x, y) instead of (y, x)

    labels = np.ones(len(points), dtype=int)
    return img, np.array(masks), np.array(points), labels


def visualize(img, masks, points):
    """
    Visualize the original image, the mask, and the mask with points.
    """
    plt.figure(figsize=(15, 5))

    # Show the original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    # Show the mask only
    plt.subplot(1, 3, 2)
    if len(masks) > 0:
        plt.imshow(masks[0] * 255, cmap="gray")
    plt.title("Mask Only")
    plt.axis("off")

    # Show the mask with points
    plt.subplot(1, 3, 3)
    if len(masks) > 0:
        plt.imshow(masks[0] * 255, cmap="gray")
    for point in points:
        if len(point) > 1:  # Check if the point has both x and y coordinates
            plt.plot(point[0], point[1], "ro")  # Mark the points in red
    plt.title("Mask with Points")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Run the read_batch function to get an image, mask, and points
# img, masks, points, _ = read_batch(data)

# Visualize the results
# visualize(img, masks, points)

checkpoint = "./sam2.1_hiera_small.pt"
model_cfg = "/configs/sam2.1/sam2.1_hiera_s.yaml"
sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")  # load model
predictor = SAM2ImagePredictor(sam2_model)  # load net


predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder


optimizer = torch.optim.AdamW(
    params=predictor.model.parameters(), lr=1e-5, weight_decay=4e-5
)

scaler = torch.amp.GradScaler()  # set mixed precision


for itr in range(100000):
    with torch.amp.autocast("cuda"):  # cast to mix precision
        image, mask, input_point, input_label = read_batch(data)  # load data batch
        if mask.shape[0] == 0:
            continue  # ignore empty batches
        predictor.set_image(image)  # apply SAM image encoder to the image

        # prompt encoding

        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point, input_label, box=None, mask_logits=None, normalize_coords=True
        )

        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels),
            boxes=None,
            masks=None,
        )

        # mask decoder

        batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
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
        )  # Upscale the masks to the original image resolution

        # Segmentaion Loss caclulation

        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])  # Turn logit map to probability map
        seg_loss = (
            -gt_mask * torch.log(prd_mask + 0.00001)
            - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)
        ).mean()  # cross entropy loss

        # Score loss calculation (intersection over union) IOU

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.05  # mix losses

        # apply back propogation

        predictor.model.zero_grad()  # empty gradient
        scaler.scale(loss).backward()  # Backpropogate
        scaler.step(optimizer)
        scaler.update()  # Mix precision

        if itr % 1000 == 0:
            torch.save(predictor.model.state_dict(), "model.torch")
            print("save model")

        # Display results

        if itr == 0:
            mean_iou = 0
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        print("step)", itr, "Accuracy(IOU)=", mean_iou)
