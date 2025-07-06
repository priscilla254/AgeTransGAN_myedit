import os
import argparse
from PIL import Image
import lm_process
import img_process
import age_transfer
import torch
from torchvision import transforms
import numpy as np
import cv2
import csv
import time
import pandas as pd

print("🔧 Importing necessary libraries...")

def main(args):
    print("🔧 Inside main()")

    # === Load seed image metadata ===
    seed_metadata_path = os.path.join(args.dir, "metadata.csv")
    if not os.path.isfile(seed_metadata_path):
        print(f"❌ Could not find metadata.csv in {args.dir}")
        return

    seed_metadata = pd.read_csv(seed_metadata_path)
    metadata_map = dict(zip(seed_metadata["filename"], seed_metadata["ethnicity"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    print("📌 Initializing model components...")
    LMP = lm_process.LandmarkProcessing()
    IMP = img_process.ImageProcessing()

    print("📥 Creating model...")
    model = age_transfer.Model(args)
    model.G_model.to(device)
    print("✅ Model created")

    if args.snapshot and os.path.isfile(args.snapshot):
        print(f"📥 Loading model checkpoint from {args.snapshot}")
        checkpoint = torch.load(args.snapshot, map_location=device)
        model.G_model.load_state_dict(checkpoint['g_ema'])
        print("✅ Model checkpoint loaded")
    else:
        print(f"⚠️ No valid checkpoint found at {args.snapshot}, using uninitialized model")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Collect input images
    image_files = []
    image_extensions = ['.jpg', '.jpeg', '.png']
    if os.path.isdir(args.dir):
        image_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir)
                       if os.path.splitext(f)[1].lower() in image_extensions]
    elif os.path.isfile(args.dir) and os.path.splitext(args.dir)[1].lower() in image_extensions:
        image_files = [args.dir]
    else:
        print(f"❌ Provided path is not a valid image file or directory: {args.dir}")
        return

    print(f"📂 Found {len(image_files)} images")
    print(f"🎯 Generating {args.group} age groups per image")

    for idx, image_path in enumerate(image_files):
        print(f"\n🖼️ [{idx+1}/{len(image_files)}] Processing {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Skipping unreadable image: {image_path}")
            continue

        print("🔍 Detecting landmarks...")
        lm = LMP.detector(img)
        if lm is None:
            print(f"⚠️ Landmark detection failed: {image_path}")
            continue

        cropped = IMP.crop(img, lm)
        filename_base = os.path.splitext(os.path.basename(image_path))[0]
        cropped = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        cropped_tensor = transform(cropped).unsqueeze(0).to(device)

        print("🎨 Generating age-transformed images...")
        start_time = time.time()
        clean_images = model.generate_image(
            cropped_tensor,
            args.group,
            return_all=True,
            interpolate=False
        )
        elapsed = time.time() - start_time
        print(f"⏱ Generation time: {elapsed:.2f} seconds")

        if clean_images is None:
            print("❌ Model returned no images. Skipping.")
            continue

        # Get ethnicity from metadata
        original_filename = os.path.basename(image_path)
        ethnicity = metadata_map.get(original_filename, "Unknown")
        ethnicity_dir = os.path.join("step2_seed_images", ethnicity)
        os.makedirs(ethnicity_dir, exist_ok=True)
        metadata_path = os.path.join(ethnicity_dir, "metadata.csv")
        metadata_rows = []

        for age_idx, img_tensor in enumerate(clean_images):
            img = transforms.ToPILImage()((img_tensor.cpu().clamp(-1, 1) + 1) / 2)
            filename = f"{filename_base}_age{age_idx + 1}.png"
            save_path = os.path.join(ethnicity_dir, filename)
            img.save(save_path)
            print(f"💾 Saved: {save_path}")

            metadata_rows.append({
                "filename": filename,
                "original_image": original_filename,
                "ethnicity": ethnicity,
                "age_group": f"age{age_idx + 1}"
            })

        # Append to or create metadata.csv
        if os.path.exists(metadata_path):
            with open(metadata_path, "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "original_image", "ethnicity", "age_group"])
                writer.writerows(metadata_rows)
        else:
            with open(metadata_path, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["filename", "original_image", "ethnicity", "age_group"])
                writer.writeheader()
                writer.writerows(metadata_rows)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=1024, help='cropping size')
    parser.add_argument('--dir', type=str, default=None, help='Directory containing images or a single image file')
    parser.add_argument('--group', type=int, default=10, help='Number of age groups (e.g., 4 or 10)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--snapshot', type=str, default='./snapshot/140000.pt', help='Path to model checkpoint')
    args = parser.parse_args()
    main(args)
