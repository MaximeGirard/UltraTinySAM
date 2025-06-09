import os
import torch
import random
import logging
import numpy as np
from copy import deepcopy
from PIL import Image as PILImage
from pycocotools.coco import COCO

from torchvision.datasets.vision import VisionDataset

from training.utils.data_utils import Frame, Object, VideoDatapoint

MAX_RETRIES = 100


class COCOSegmentationDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        ann_file: str,
        transforms=None,
        training=True,
        multiplier=1,
        always_target=True,
        target_segments_available=True,
        max_num_objects=3,  # Maximum number of objects to sample
    ):
        super().__init__(root, transforms)
        self._transforms = transforms
        self.training = training
        self.coco = COCO(ann_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        self.repeat_factors = torch.ones(len(self.image_ids), dtype=torch.float32) * multiplier
        self.curr_epoch = 0
        self.always_target = always_target
        self.target_segments_available = target_segments_available
        self.max_num_objects = max_num_objects

        print(f"COCO dataset loaded with {len(self.image_ids)} images.")

    def _get_datapoint(self, idx):
        for retry in range(MAX_RETRIES):
            try:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()

                image_id = self.image_ids[idx]
                image_info = self.coco.imgs[image_id]
                image_path = os.path.join(self.root, image_info["file_name"])

                ann_ids = self.coco.getAnnIds(imgIds=image_id)
                anns = self.coco.loadAnns(ann_ids)

                image = self._load_image(image_path)
                masks = self._load_masks(anns, image_info)
                mask_keys = list(masks.keys())

                if len(masks) > self.max_num_objects:
                    # Randomly select max_num_objects masks
                    selected_ids = random.sample(mask_keys, self.max_num_objects)
                    masks = {obj_id: masks[obj_id] for obj_id in selected_ids}

                elif len(masks) < self.max_num_objects:
                    if len(masks) == 0:
                        # If there are no masks, fill with None
                        masks = {i: torch.zeros((image_info["height"], image_info["width"]), dtype=torch.uint8)
                                 for i in range(self.max_num_objects)}
                    else:
                        # Duplicate masks randomly until we reach max_num_objects
                        additional_ids = random.choices(mask_keys, k=self.max_num_objects - len(masks))
                        next_id = max(mask_keys) + 1 if mask_keys else 1  # Ensure numeric IDs

                        for obj_id in additional_ids:
                            masks[next_id] = masks[obj_id].clone()
                            next_id += 1  # Keep IDs unique

                datapoint = self.construct(image, masks, image_id)

                if self._transforms:
                    for transform in self._transforms:
                        datapoint = transform(datapoint, epoch=self.curr_epoch)

                return datapoint

            except Exception as e:
                if self.training:
                    logging.warning(f"Loading failed (id={idx}); Retry {retry} with exception: {e}")
                    idx = random.randrange(0, len(self.image_ids))
                else:
                    raise e

    def construct(self, image, masks, image_id):
        w, h = image.size
        frame = Frame(data=image, objects=[])

        for obj_id, mask in masks.items():
            frame.objects.append(Object(object_id=obj_id, frame_index=0, segment=mask))

        return VideoDatapoint(frames=[frame], video_id=image_id, size=(h, w))

    def _load_image(self, image_path):
        with open(image_path, "rb") as fopen:
            return PILImage.open(fopen).convert("RGB")

    def _load_masks(self, anns, image_info):
        h, w = image_info["height"], image_info["width"]
        masks = {}

        for ann in anns:
            if "segmentation" not in ann or not ann["segmentation"]:
                continue

            mask = self.coco.annToMask(ann)
            masks[ann["id"]] = torch.from_numpy(mask)
        
        return masks

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.image_ids)
