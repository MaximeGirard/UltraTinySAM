import argparse
import json
import os
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm


# PARAMETERS
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess SA-1B dataset tars with resizing and filtering.")
    parser.add_argument('--target-dir', type=str, default='../datasets/SA1B_preprocessed_up_150',
                        help='Directory to save preprocessed images and annotations.')
    parser.add_argument('--temp-extract-dir', type=str, default='../datasets/temp_extract',
                        help='Temporary directory for extracting tar files.')
    parser.add_argument('--image-size', type=int, nargs=2, default=[128, 128],
                        help='Output image size as two integers: width height.')
    parser.add_argument('--area-min', type=int, default=150,
                        help='Minimum area for valid masks. Default is 150.')
    parser.add_argument('--num-workers', type=int, default=12,
                        help='Number of parallel workers.')
    parser.add_argument('--to-download-file', type=str, default='../datasets/to_download.txt',
                        help='Path to file listing tar files to download and process.')
    return parser.parse_args()

args = parse_args()
TARGET_DIR = args.target_dir
TEMP_EXTRACT_DIR = args.temp_extract_dir
IMAGE_SIZE = tuple(args.image_size)
AREA_MIN = args.area_min
NUM_WORKERS = args.num_workers
TO_DOWNLOAD_FILE = args.to_download_file

os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(TEMP_EXTRACT_DIR, exist_ok=True)

def download_file(url, dest):
    print(f"⬇️ Downloading {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"✅ Download complete: {dest}")

def extract_tar(tar_path, extract_to):
    print(f"📦 Extracting {tar_path}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_to)
    print(f"✅ Extracted to {extract_to}")

def resize_rle(rle, original_size, new_size):
    mask = mask_utils.decode(rle)
    mask_resized = np.array(Image.fromarray(mask).resize(new_size[::-1], resample=Image.NEAREST))
    rle_resized = mask_utils.encode(np.asfortranarray(mask_resized))
    rle_resized['size'] = [new_size[0], new_size[1]]
    return rle_resized, mask_resized

def process_sample(image_path, json_path, output_image_path, output_json_path):
    try:
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        original_w, original_h = original_size

        with open(json_path, 'r') as f:
            data = json.load(f)

        new_annotations = []
        for ann in data['annotations']:
            rle = ann['segmentation']
            rle['counts'] = rle['counts'].encode('utf-8') if isinstance(rle['counts'], str) else rle['counts']
            resized_rle, resized_mask = resize_rle(rle, (original_h, original_w), (IMAGE_SIZE[1], IMAGE_SIZE[0]))

            area = int(resized_mask.sum())
            if area < AREA_MIN:
                continue

            y_indices, x_indices = np.where(resized_mask)
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue

            x_min = int(np.min(x_indices))
            y_min = int(np.min(y_indices))
            x_max = int(np.max(x_indices))
            y_max = int(np.max(y_indices))
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            ann['bbox'] = bbox
            ann['area'] = area
            resized_rle['counts'] = resized_rle['counts'].decode('utf-8')
            ann['segmentation'] = resized_rle

            new_annotations.append(ann)

        if len(new_annotations) == 0:
            print(f"⚠️ Skipping {image_path} — no valid annotations left.")
            return

        image_resized = image.resize(IMAGE_SIZE, resample=Image.BILINEAR)
        os.makedirs(output_image_path.parent, exist_ok=True)
        image_resized.save(output_image_path)

        data['image']['width'] = IMAGE_SIZE[0]
        data['image']['height'] = IMAGE_SIZE[1]
        data['image']['file_name'] = output_image_path.name
        data['annotations'] = new_annotations

        os.makedirs(output_json_path.parent, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")

def process_image(image_path_output_dir_temp_extract):
    image_path, output_dir, temp_extract_dir = image_path_output_dir_temp_extract
    json_path = image_path.with_suffix('.json')
    rel_path = image_path.relative_to(temp_extract_dir)
    output_image_path = output_dir / rel_path
    output_json_path = output_image_path.with_suffix('.json')

    if json_path.exists():
        process_sample(image_path, json_path, output_image_path, output_json_path)

def process_tar(tar_name, tar_url):
    tar_path = Path(TEMP_EXTRACT_DIR) / tar_name
    output_dir = Path(TARGET_DIR) / tar_name.replace('.tar', '')

    download_file(tar_url, tar_path)
    extract_tar(tar_path, TEMP_EXTRACT_DIR)

    image_files = list(Path(TEMP_EXTRACT_DIR).rglob('*.jpg'))
    args = [(image_path, output_dir, TEMP_EXTRACT_DIR) for image_path in image_files]

    print(f"⚙️ Processing {len(image_files)} images in parallel with {NUM_WORKERS} workers into {output_dir}")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(process_image, args), total=len(image_files)))

    print(f"🧹 Cleaning up tar and temporary files...")
    tar_path.unlink(missing_ok=True)
    shutil.rmtree(TEMP_EXTRACT_DIR)
    os.makedirs(TEMP_EXTRACT_DIR, exist_ok=True)  # recreate for next tar

def main():
    with open(TO_DOWNLOAD_FILE, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split('\t')
        if len(parts) != 2:
            print(f"⚠️ Skipping invalid line: {line}")
            continue
        tar_name, tar_url = parts
        print(f"\n=== Processing tar: {tar_name} ===")
        process_tar(tar_name, tar_url)

    print("\n✅ All tar files processed!")

if __name__ == "__main__":
    main()