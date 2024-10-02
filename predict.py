import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# use bfloat16 for the entire script (memory efficient)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()


image_path = r"C:\Users\jules\OneDrive\Bureau\nails\index_nail_3.jpg"
mask_path = (
    r"sample_mask.png"  # path to mask, the mask will define the image region to segment
)


def read_image(image_path, mask_path):  # read and resize image and mask
    img = cv2.imread(image_path)[..., ::-1]  # read image as rgb

    # Resize image to maximum size of 1024

    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    return img


image = read_image(image_path, mask_path)


# Load model you need to have pretrained model already made
checkpoint = "./sam2.1_hiera_small.pt"
model_cfg = "/configs/sam2.1/sam2.1_hiera_s.yaml"
sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")  # load model
predictor = SAM2ImagePredictor(sam2_model)  # load net

predictor.model.load_state_dict(torch.load("model.torch"))


with torch.no_grad():  # prevent the net from caclulate gradient (more efficient inference)
    predictor.set_image(image)  # image encoder
    masks, scores, logits = predictor.predict(  # prompt encoder + mask decoder
        point_coords=[
            [image.shape[1] // 2, image.shape[0] // 2]
        ],  # center of the image
        point_labels=np.ones(1, dtype=int),
    )


masks = masks.astype(bool)
sorted_indices = np.argsort(scores)[::-1]  # Sort scores in descending order
sorted_masks = masks[sorted_indices].astype(bool)

seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)


for i in range(sorted_masks.shape[0]):
    mask = sorted_masks[i]
    if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
        continue
    mask[occupancy_mask] = 0
    seg_map[mask] = i + 1
    occupancy_mask[mask] = 1


rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
for id_class in range(1, seg_map.max() + 1):
    rgb_image[seg_map == id_class] = [
        np.random.randint(255),
        np.random.randint(255),
        np.random.randint(255),
    ]


# Create mixed image
mixed_image = (rgb_image / 2 + image / 2).astype(np.uint8)

# Plot using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(mixed_image)
axes[1].set_title("Mixed Image")
axes[1].axis("off")

plt.show()
