import os

# Path to your COCO annotation file
# coco_annotation_path = r"C:\Users\jules\Downloads\nails_segmentation.v44i.coco-segmentation\train\_annotations.coco.json"

# # Load COCO annotations
# with open(coco_annotation_path, "r") as f:
#     coco = json.load(f)

# for image in tqdm(coco["images"]):
#     img_id = image["id"]
#     img_filename = image["file_name"]
#     if not os.path.exists(os.path.join("data", "mask_images", f"{img_id}.jpg")):
#         continue

#     # find the image in the images directory
#     images_dir = (
#         r"C:\Users\jules\Downloads\nails_segmentation.v44i.coco-segmentation\train"
#     )
#     if os.path.exists(os.path.join(images_dir, img_filename)):
#         # copy the image to the data directory with the id as the filename
#         shutil.copy(
#             os.path.join(images_dir, img_filename),
#             os.path.join("data", "images", f"{img_id}.jpg"),
#         )


mask_names = set(os.listdir("./data/mask_images"))

# Path to the directory containing the images
image_names = set(os.listdir("./data/images"))

print(f"Number of mask images: {len(mask_names)}")
print(f"Number of images: {len(image_names)}")

names_in_mask_not_in_images = mask_names - image_names

print(f"Number of mask images not in images: {len(names_in_mask_not_in_images)}")

names_in_images_not_in_mask = image_names - mask_names

print(f"Number of images not in mask images: {len(names_in_images_not_in_mask)}")

# remove the mask images that don't have corresponding images
# for name in names_in_mask_not_in_images:
#     os.remove(os.path.join("./data/mask_images", name))
