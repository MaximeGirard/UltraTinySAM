# UltraTinySAM export script for IMX500 deployment
# Made by: Maxime Girard, 2025

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil
import json
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import mct_quantizers as mctq
from collections import defaultdict


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SAM models on COCO/LVIS datasets")
    
    # Dataset arguments
    parser.add_argument("--img-dir", default="datasets/COCO/val2017", help="Path to COCO/LVIS images directory")
    parser.add_argument("--ann-file", default="datasets/COCO/annotations/instances_val2017.json", help="Path to COCO/LVIS annotations file")
    
    # Model arguments
    parser.add_argument("--fp-model", help="Path to floating-point ONNX model")
    parser.add_argument("--quantized-model", help="Path to quantized ONNX model")
    
    # Evaluation parameters
    parser.add_argument("--input-size", type=int, default=128, help="Input image size")
    parser.add_argument("--max-images", type=int, default=100, help="Maximum number of images to evaluate")
    parser.add_argument("--min-area", type=int, default=150, help="Minimum mask area threshold")
    parser.add_argument("--click-mode", choices=["1-click", "3-clicks", "5-clicks", "gt-box"], 
                       default="1-click", help="Click mode for model input")
    
    # Visualization arguments
    parser.add_argument("--visualize", action="store_true", help="Generate visualization outputs")
    parser.add_argument("--display-only-best-mask", action="store_true", 
                       help="Display only the best mask in visualizations")
    
    return parser.parse_args()

args = parse_arguments()

# Validate arguments
if not args.fp_model and not args.quantized_model:
    raise ValueError("At least one model (--fp-model or --quantized-model) must be provided")

# Setup paths and parameters
COCO_IMG_DIR = args.img_dir
COCO_ANN_FILE = args.ann_file
INPUT_SIZE = args.input_size
MAX_IMAGES = args.max_images
MIN_AREA = args.min_area
CLICK_MODE = args.click_mode
VISUALIZE = args.visualize
DISPLAY_ONLY_BEST_MASK = args.display_only_best_mask
VISUALIZE_DIR = "visualizations_" + CLICK_MODE

USE_FP = args.fp_model is not None
USE_QUANTIZED = args.quantized_model is not None

ONNX_MODEL_PATH = args.fp_model if USE_FP else None
ONNX_MODEL_Q_PATH = args.quantized_model if USE_QUANTIZED else None

transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

if USE_FP:
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
if USE_QUANTIZED:
    session_options = mctq.get_ort_session_options()
    session_options.enable_mem_reuse = False
    ort_session_q = ort.InferenceSession(ONNX_MODEL_Q_PATH, session_options, providers=["CPUExecutionProvider"])


def get_image_filename_from_lvis(img_info):
    """Extract filename from LVIS coco_url field"""
    if 'file_name' in img_info:
        return img_info['file_name']
    elif 'coco_url' in img_info:
        return img_info['coco_url'].split('/')[-1]
    else:
        return f"{img_info['id']:012d}.jpg"


def visualize_multiple_predictions_vs_gt(img, gt_masks, pred_masks, point_coords_list, iou_predictions_list, save_path="vis_output.png", click_mode="gt-box"):
    img_np = np.array(img)
    vis_img = img_np.copy()

    for gt_mask in gt_masks:
        gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, gt_contours, -1, (255, 255, 0), 1)

    for pred_set in pred_masks:
        if len(pred_set) == 1:
            mask = pred_set[0]
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, (255, 0, 0), 1)
        else:
            for idx, mask in reversed(list(enumerate(pred_set))):
                color = (255, 0, 0) if idx == 0 else (80, 50, 150)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, color, 1)

    for coords in point_coords_list:
        if click_mode == "gt-box" and len(coords) == 2:
            x1, y1 = map(int, coords[0])
            x2, y2 = map(int, coords[1])
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (128, 128, 0), 1)
        else:
            for x, y in coords:
                cv2.drawMarker(vis_img, (int(x), int(y)), (238, 210, 2), markerType=cv2.MARKER_STAR, markerSize=6, thickness=1)

    plt.figure(figsize=(6, 6))
    plt.imshow(vis_img)
    plt.axis("off")
    ious_str = ", ".join([f"{iou:.2f}" for iou in iou_predictions_list])
    plt.title(f"GT (Yellow) | Pred: Red=Best, Gray=Others\nIoU: {ious_str}")
    plt.savefig(save_path)
    plt.close()


def run_model(img_pil, coords, labels):
    img_tensor = transform(img_pil).unsqueeze(0).numpy()
    point_coords = np.array([coords], dtype=np.float32)
    point_labels = np.array([labels], dtype=np.float32)

    result = {}

    if USE_FP:
        outputs = ort_session.run(None, {
            "image": img_tensor,
            "point_coords": point_coords,
            #"point_labels": point_labels => now fixed in the model
        })
        masks, iou_predictions = outputs
        masks = F.interpolate(torch.from_numpy(masks), size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False).numpy()
        binary_masks = [(masks[0][idx] > 0).astype(np.uint8) for idx in range(3)]
        result["fp"] = (binary_masks, iou_predictions[0])

    if USE_QUANTIZED:
        outputs_q = ort_session_q.run(None, {
            "input": img_tensor,
            "inputs.3": point_coords,
        })
        masks_q, iou_predictions_q = outputs_q
        masks_q = F.interpolate(torch.from_numpy(masks_q), size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False).numpy()
        binary_masks_q = [(masks_q[0][idx] > 0).astype(np.uint8) for idx in range(3)]
        result["quantized"] = (binary_masks_q, iou_predictions_q[0])

    return result


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0.0

def compute_ap(tp, fp, n_gt):
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / n_gt
    prec = tp_cum / (tp_cum + fp_cum + 1e-6)

    # 11-point interpolation (legacy VOC-style)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        prec_at_t = prec[rec >= t]
        p = np.max(prec_at_t) if prec_at_t.size > 0 else 0
        ap += p / 11.0
    return ap

def manual_eval(predictions, coco_gt, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    aps = []

    for iou_thresh in iou_thresholds:
        image_to_gts = defaultdict(list)
        image_to_preds = defaultdict(list)

        # Organize GTs
        for img_id in coco_gt.getImgIds():
            anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
            for ann in anns:
                mask = coco_gt.annToMask(ann).astype(bool)
                if mask.sum() >= MIN_AREA:
                    image_to_gts[img_id].append({
                        "category_id": ann["category_id"],
                        "used": False,
                        "mask": mask
                    })

        # Organize predictions
        for pred in predictions:
            img_id = pred["image_id"]
            mask = mask_utils.decode(pred["segmentation"]).astype(bool)
            image_to_preds[img_id].append({
                "category_id": pred["category_id"],
                "mask": mask,
                "score": pred["score"]
            })

        tp_list = []
        fp_list = []
        total_gt = 0

        for img_id in image_to_preds.keys():
            preds = sorted(image_to_preds[img_id], key=lambda x: -x["score"])
            gts = image_to_gts.get(img_id, [])

            total_gt += len(gts)

            for pred in preds:
                matched = False
                for gt in gts:
                    if gt["used"]:
                        continue
                    if pred["category_id"] != gt["category_id"]:
                        continue
                    iou = compute_iou(pred["mask"], gt["mask"])
                    if iou >= iou_thresh:
                        matched = True
                        gt["used"] = True
                        break
                if matched:
                    tp_list.append(1)
                    fp_list.append(0)
                else:
                    tp_list.append(0)
                    fp_list.append(1)

        ap = compute_ap(np.array(tp_list), np.array(fp_list), total_gt)
        aps.append(ap)
        print(f"AP@{iou_thresh:.2f}: {ap:.4f}")

    mean_ap = np.mean(aps)
    print(f"\nManual mAP: {mean_ap:.4f} over {total_gt} instances")


# Setup preprocessing
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Setup COCO evaluation
coco = COCO(COCO_ANN_FILE)
image_ids = coco.getImgIds()
image_ids.sort()
print(f"Total images in COCO: {len(image_ids)}")
image_ids = image_ids[:MAX_IMAGES]
print(f"Using {len(image_ids)} images for evaluation.")

predictions_fp = []
predictions_quant = []

if VISUALIZE:
    shutil.rmtree(VISUALIZE_DIR, ignore_errors=True)
    os.makedirs(VISUALIZE_DIR, exist_ok=True)

for img_id in tqdm(image_ids):
    img_info = coco.loadImgs(img_id)[0]
    
    image_url = img_info.get('coco_url', None)
    
    # Get the correct filename for LVIS
    filename = get_image_filename_from_lvis(img_info)
    img_path = os.path.join(COCO_IMG_DIR, filename)

    img = Image.open(img_path).convert("RGB")
    width, height = img.size

    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    if len(anns) == 0:
        continue

    for ann in anns:
        gt_mask = coco.annToMask(ann)
        gt_mask_resized = cv2.resize(gt_mask, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)

        if np.sum(gt_mask_resized) < MIN_AREA:
            continue

        y_idxs, x_idxs = np.where(gt_mask == 1)
        if len(x_idxs) == 0:
            continue

        if CLICK_MODE == "gt-box":
            x, y, w, h = ann['bbox']
            coords_input = [
                [x * INPUT_SIZE / width, y * INPUT_SIZE / height],
                [(x + w) * INPUT_SIZE / width, (y + h) * INPUT_SIZE / height]
            ]
            labels_input = [2, 3]
        else:
            num_clicks = {"1-click": 1, "3-clicks": 3, "5-clicks": 5}[CLICK_MODE]

            # Compute center of the mask (center of mass)
            mask_center_y, mask_center_x = np.mean(y_idxs), np.mean(x_idxs)

            # Rescale to input size
            center_input_x = mask_center_x * INPUT_SIZE / width
            center_input_y = mask_center_y * INPUT_SIZE / height

            coords_input = [[center_input_x, center_input_y]]
            labels_input = [1]

            if num_clicks > 1:
                # Get all candidate indices
                all_indices = list(range(len(x_idxs)))

                # Identify index closest to the center to avoid duplication
                distances = np.sqrt((x_idxs - mask_center_x)**2 + (y_idxs - mask_center_y)**2)
                center_index = np.argmin(distances)

                # Remove the center index from candidates
                all_indices.remove(center_index)

                if len(all_indices) < (num_clicks - 1):
                    continue  # Not enough points to sample from

                # Randomly sample the remaining points
                selected_indices = np.random.choice(all_indices, num_clicks - 1, replace=False)

                for i in selected_indices:
                    cx = x_idxs[i] * INPUT_SIZE / width
                    cy = y_idxs[i] * INPUT_SIZE / height
                    coords_input.append([cx, cy])
                    labels_input.append(1)

        model_outputs = run_model(img, coords_input, labels_input)
    
        if "fp" in model_outputs:
            masks_fp, iou_fp = model_outputs["fp"]
            best_idx_fp = np.argmax(iou_fp)
            best_mask_fp = masks_fp[best_idx_fp]
            pred_mask_resized_fp = cv2.resize(best_mask_fp, (width, height), interpolation=cv2.INTER_NEAREST)
            rle_fp = mask_utils.encode(np.asfortranarray(pred_mask_resized_fp.astype(np.uint8)))
            rle_fp["counts"] = rle_fp["counts"].decode("utf-8")
            predictions_fp.append({
                "image_id": img_id,
                "category_id": ann["category_id"],
                "segmentation": rle_fp,
                "score": 1.0
            })
            
        if "quantized" in model_outputs:
            masks_q, iou_q = model_outputs["quantized"]
            best_idx_q = np.argmax(iou_q)
            best_mask_q = masks_q[best_idx_q]
            pred_mask_resized_q = cv2.resize(best_mask_q, (width, height), interpolation=cv2.INTER_NEAREST)
            rle_q = mask_utils.encode(np.asfortranarray(pred_mask_resized_q.astype(np.uint8)))
            rle_q["counts"] = rle_q["counts"].decode("utf-8")
            predictions_quant.append({
                "image_id": img_id,
                "category_id": ann["category_id"],
                "segmentation": rle_q,
                "score": 1.0
            })

        if VISUALIZE:
            img_resized = img.resize((INPUT_SIZE, INPUT_SIZE))
            gt_masks_list = [gt_mask_resized]
            coords_list = [coords_input]

            if "fp" in model_outputs:
                pred_masks = [masks_fp[best_idx_fp]] if DISPLAY_ONLY_BEST_MASK else [masks_fp[best_idx_fp]] + [m for i, m in enumerate(masks_fp) if i != best_idx_fp]
                visualize_multiple_predictions_vs_gt(
                    img_resized, gt_masks_list, [pred_masks], coords_list,
                    [float(np.max(iou_fp))],
                    save_path=os.path.join(VISUALIZE_DIR, f"img{img_id}_{ann['id']}_fp.jpg")
                )

            if "quantized" in model_outputs:
                masks_q, iou_q = model_outputs["quantized"]
                best_idx_q = np.argmax(iou_q)
                pred_masks_q = [masks_q[best_idx_q]] if DISPLAY_ONLY_BEST_MASK else [masks_q[best_idx_q]] + [m for i, m in enumerate(masks_q) if i != best_idx_q]
                visualize_multiple_predictions_vs_gt(
                    img_resized, gt_masks_list, [pred_masks_q], coords_list,
                    [float(np.max(iou_q))],
                    save_path=os.path.join(VISUALIZE_DIR, f"img{img_id}_{ann['id']}_q.jpg")
                )

# Save predictions
with open("predictions_fp.json", "w") as f:
    json.dump(predictions_fp, f)

with open("predictions_quant.json", "w") as f:
    json.dump(predictions_quant, f)

# Evaluate
if USE_FP:
    with open("predictions_fp.json", "r") as f:
        predictions_fp = json.load(f)

    print("\n=== [Manual Evaluation for FP Model] ===")
    manual_eval(predictions_fp, coco)

if USE_QUANTIZED:
    with open("predictions_quant.json", "r") as f:
        predictions_quant = json.load(f)

    print("\n=== [Manual Evaluation for Quantized Model] ===")
    manual_eval(predictions_quant, coco)