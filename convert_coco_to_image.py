import json
import os

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# Path to your COCO annotation file
coco_annotation_path = r"C:\Users\jules\Downloads\nails_segmentation.v44i.coco-segmentation\train\_annotations.coco.json"

# Load COCO annotations
with open(coco_annotation_path, "r") as f:
    coco = json.load(f)

# Create directories for mask output
mask_output_dir = "data/mask_images"
os.makedirs(mask_output_dir, exist_ok=True)

# Create a dictionary to map image IDs to file names and dimensions
img_id_to_filename = {
    img["id"]: (img["file_name"], img["width"], img["height"]) for img in coco["images"]
}

# Group annotations by image ID
annotations_by_image = {}
for annotation in coco["annotations"]:
    img_id = annotation["image_id"]
    if img_id not in annotations_by_image:
        annotations_by_image[img_id] = []
    annotations_by_image[img_id].append(annotation)

# Process each image and generate a mask with all its annotations
for img_id, annotations in tqdm(annotations_by_image.items()):
    img_filename, img_width, img_height = img_id_to_filename[img_id]

    # Create an empty mask with the correct dimensions for the image
    mask = Image.new(
        "L", (img_width, img_height), 0
    )  # 'L' mode for single-channel (0-255 values)

    # Draw all the polygons for this image
    for annotation in annotations:
        segmentation = annotation["segmentation"]
        for seg in segmentation:
            polygon = [(int(seg[i]), int(seg[i + 1])) for i in range(0, len(seg), 2)]

            # Ensure polygon points are within the image dimensions
            for i, point in enumerate(polygon):
                x, y = point
                polygon[i] = (
                    min(max(x, 0), img_width - 1),
                    min(max(y, 0), img_height - 1),
                )

            # Draw the polygon on the mask
            ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)

    # Convert the mask to an array and save as .png
    mask = np.array(mask) * 255  # Multiply by 255 to get 0/255 values
    mask_img = Image.fromarray(mask)

    # Save with a simplified filename
    simple_filename = f"{img_id}.jpg"
    mask_img.save(os.path.join(mask_output_dir, simple_filename))

print("Masks have been successfully saved in", mask_output_dir)
