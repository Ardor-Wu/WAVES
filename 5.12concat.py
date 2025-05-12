import os
import re
import torch
from torchvision import transforms
from PIL import Image

# Input directories with output filenames
input_configs = [
    (
        "/scratch/qilong3/WAVES/data/out/strength_12.75",
        "/scratch/qilong3/WAVES/processed_images_strength_12.75.pt"
    ),
    (
        "/scratch/qilong3/WAVES/data/out/strength_31.875",
        "/scratch/qilong3/WAVES/processed_images_strength_31.875.pt"
    ),
    (
        "/scratch/qilong3/WAVES/data/out/strength_12",
        "/scratch/qilong3/WAVES/processed_images_strength_12.pt"
    ),
    (
        "/scratch/qilong3/WAVES/data/out/strength_31",
        "/scratch/qilong3/WAVES/processed_images_strength_31.pt"
    ),
]

# Filename pattern: extract batch and idx
pattern = re.compile(r"image_batch(\d+)_idx(\d+)\.png")

# Transformation: to tensor and normalize to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Process each directory separately
for input_dir, output_path in input_configs:
    print(f"Processing {input_dir} → {output_path}")
    image_files = []

    for fname in os.listdir(input_dir):
        match = pattern.match(fname)
        if match:
            batch = int(match.group(1))
            idx = int(match.group(2))
            image_files.append(((batch, idx), os.path.join(input_dir, fname)))

    # Sort files by (batch, idx)
    image_files.sort(key=lambda x: (x[0][0], x[0][1]))

    # Load and transform images
    tensor_list = []
    for (_, path) in image_files:
        img = Image.open(path).convert("RGB")
        tensor = transform(img)
        tensor_list.append(tensor)

    # Stack and save
    final_tensor = torch.stack(tensor_list)
    torch.save(final_tensor, output_path)
    print(f"Saved {final_tensor.shape} to {output_path}")
