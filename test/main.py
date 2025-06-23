import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import lm_process
import img_process
import age_transfer
from torch.autograd import Variable
import torch
from torchvision import transforms
import numpy as np
import cv2
from datetime import datetime
from make_video import make_video, play_video
import torchvision.utils as vutils
import time

print("🔧 Importing necessary libraries...")

def annotate_image(img, label, font_size=24):
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    padding = 5
    x, y = padding, padding

    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill="black"
    )
    draw.text((x, y), label, font=font, fill="white")

    return img

def main(args):
    print("🔧 Inside main()")
    
    os.makedirs("annotated_results", exist_ok=True)

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

    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    if args.dir is not None:
        if os.path.isdir(args.dir):
            image_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir)
                           if os.path.splitext(f)[1].lower() in image_extensions]
        elif os.path.isfile(args.dir) and os.path.splitext(args.dir)[1].lower() in image_extensions:
            image_files = [args.dir]
        else:
            print(f"❌ Provided path is not a valid image file or directory: {args.dir}")
            return
    else:
        print("❌ Please provide a directory or image file with --dir")
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

        print("✂️ Cropping image around landmarks")
        cropped = IMP.crop(img, lm)
        filename_base = os.path.splitext(os.path.basename(image_path))[0]
        cropped = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        print("📦 Converting to tensor")
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

        for age_idx, img_tensor in enumerate(clean_images):
            img = transforms.ToPILImage()((img_tensor.cpu().clamp(-1, 1) + 1) / 2)
            age_label = f"Age group {age_idx + 1}"
            annotated_img = annotate_image(img, age_label)

            save_path = f"annotated_results/{filename_base}_age{age_idx + 1}.png"
            annotated_img.save(save_path)
            print(f"💾 Saved: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=1024, help='cropping size')
    parser.add_argument('--dir', type=str, default=None, help='Directory containing images or a single image file')
    parser.add_argument('--group', type=int, default=10, help='Number of age groups (e.g., 4 or 10)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--snapshot', type=str, default='./snapshot/140000.pt', help='Path to model checkpoint')

    args = parser.parse_args()
    main(args)
