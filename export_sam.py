# UltraTinySAM export script for IMX500 deployment
# Made by: Maxime Girard, 2025

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import os
import sys
from typing import Any, Optional, Tuple

import model_compression_toolkit as mct
import numpy as np
import torch
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, parameter_count
from hydra import initialize_config_module
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from torch import nn
from torch.ao.quantization import (QConfig, QuantStub, convert,
                                   float_qparams_weight_only_qconfig, prepare)
from torch.ao.quantization.observer import MinMaxObserver
from torch.nn.init import trunc_normal_
from torchvision import transforms
from training.utils.train_utils import makedir, register_omegaconf_resolvers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAM_INPUT_SIZE = 128
EMBED_DIM_DECODER = 256

def parse_args():
    parser = argparse.ArgumentParser(description="Export UltraTinySAM for IMX500")
    # config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ultratinysam/UltraTinySAM.yaml",
        help="Path to the configuration file",
    )
    
    # model checkpoint
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="checkpoints/utsam.pt",
        help="Path to the model checkpoint to load. Default is 'checkpoints/utsam.pt'.",
    )
    
    # target input
    parser.add_argument(
        "--target_input_coords",
        type=str,
        default="128,128",
        help="Target input coordinates for the model, in the format 'x,y'. Only if static point is set. Default is '128,128'.",
    )
    
    # static point
    parser.add_argument(
        "--static_point",
        action="store_true",
        help="Use a static point for the model. Only 1-click input is supported. Required for deployment on IMX500. Default is False.",
    )
    
    # static labels
    parser.add_argument(
        "--static_labels",
        action="store_true",
        help="Use static labels for the model. Required for quantization export. Default is False.",
    )
    
    # labels mode
    parser.add_argument(
        "--labels_mode",
        type=str,
        default="1-click",
        choices=["1-click", "3-clicks", "5-clicks", "gt-box"],
        help="Labels mode for the model. Default is '1-click'.",
    )
    
    # export fp
    parser.add_argument(
        "--export_fp",
        action="store_true",
        default=False,
        help="Export the full precision model. Default is False.",
    )
    
    # export quantized
    parser.add_argument(
        "--export_quant",
        action="store_true",
        default=True,
        help="Export the quantized model. Default is True.",
    )
    
    # exported quantized models
    parser.add_argument(
        "--exported_models",
        type=str,
        nargs="+",
        default=["full"],
        choices=["full", "encoder", "decoder"],
        help="List of models/part of the model to export. Default is ['full'].",
    )
    
    # quantization method
    parser.add_argument(
        "--quant_method",
        type=str,
        default="PTQ",
        choices=["PTQ", "QAT", "GPTQ"],
        help="Quantization method to use. Default is 'PTQ'. QAT only train the model with FakeQuantize blocks, then the checkpoint saved needs to be exported using PTQ for deployment.",
    )
    
    # load checkpoint before PTQ
    parser.add_argument(
        "--load_ckpt_before_ptq",
        action="store_true",
        default=False,
        help="Load the checkpoint (QAT checkpoint) before PTQ. Default is False.",
    )
    
    # QAT checkpoint
    parser.add_argument(
        "--qat_ckpt",
        type=str,
        default="qat_model/ultra-tiny-sam-qat-0.pth",
        help="Path to the QAT checkpoint to load before PTQ. Default is 'qat_model/ultra-tiny-sam-qat-0.pth'.",
    )
    
    # save dir for QAT checkpoint
    parser.add_argument(
        "--export_dir_output",
        type=str,
        default="qat_model",
        help="Directory to save the exported models. Default is 'qat_model'.",
    )
    
    # Num batches for QAT
    parser.add_argument(
        "--max_batches",
        type=int,
        default=20,
        help="Number of batches to use for QAT. Default is 20.",
    )
    
    # batch size for QAT
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size to use for QAT. Default is 32.",
    )
    
    # Num epochs for QAT
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs to use for QAT. Default is 1.",
    )
    
    # Num samples for representative data
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples to use for representative data. Default is 20.",
    )
    
    # image dir for representative data
    parser.add_argument(
        "--img_dir",
        type=str,
        default="datasets/SA1B_preprocessed_up_150/sa_000020",
        help="Directory containing images for representative data. Default is 'datasets/SA1B_preprocessed_up_150/sa_000020'.",
    )
    
    # dump the embeddings
    parser.add_argument(
        "--dump_coords_embeddings",
        action="store_true",
        default=False,
        help="Dump the coordinates embeddings to a npy file. Only works with 1-click mode. Default is False.",
    )
    
    args = parser.parse_args()
    return args

args = parse_args()

TARGET_POINT_X, TARGET_POINT_Y = map(int, args.target_input_coords.split(","))
if args.static_point:
    STATIC_POINT = True
    print(f"Using static point at ({TARGET_POINT_X}, {TARGET_POINT_Y})")
else:
    STATIC_POINT = False
    print("Using dynamic point for the model")    
    
if args.static_labels:
    STATIC_LABELS = True
    print("Using static labels for the model")
else:
    STATIC_LABELS = False
    print("Using dynamic labels for the model")
    
NUM_POINTS_PER_MODE = {
    "1-click": 1,
    "3-clicks": 3,
    "5-clicks": 5,
    "gt-box": 2,
}

if args.labels_mode in NUM_POINTS_PER_MODE:
    LABELS_MODE = args.labels_mode
    print(f"Using labels mode: {LABELS_MODE}")
else:
    raise ValueError(f"Invalid labels mode: {args.labels_mode}. Choose from {list(NUM_POINTS_PER_MODE.keys())}")

NUM_POINTS = NUM_POINTS_PER_MODE[LABELS_MODE]

IMG_DIR = args.img_dir
NUM_SAMPLES = args.num_samples  # Number of samples to use for representative data

assert args.export_fp or args.export_quant, "At least one of --export_fp or --export_quant must be set"
EXPORT_FP = args.export_fp
EXPORT_QUANT = args.export_quant

EXPORTED_MODELS = args.exported_models
assert EXPORTED_MODELS in ["full", "encoder", "decoder"], "Invalid exported models. Choose from ['full', 'encoder', 'decoder']"

QUANT_METHOD = args.quant_method

NAME_FP = f"ultra-tiny-sam-{LABELS_MODE}.onnx"  # Name of the FP model to export
NAME_FP_STATIC_PT = f"ultra-tiny-sam-static-{LABELS_MODE}.onnx"  # Name of the FP model with static point to export
NAME_FP_STATIC_LABELS = f"ultra-tiny-sam-static-labels-{LABELS_MODE}.onnx"  # Name of the FP model with static labels to export
NAME_PTQ = f"ptq-ultra-tiny-sam-{LABELS_MODE}.onnx"  # Name of the PTQ model to export
NAME_QAT = f"qat-ultra-tiny-sam-{LABELS_MODE}.onnx"  # Name of the QAT model to export

LOAD_CKPT_BEFORE_PTQ = args.load_ckpt_before_ptq
QAT_CKPT = args.qat_ckpt
MAX_BATCHES = args.max_batches  # Number of batches to use for QAT
NUM_EPOCHS = args.num_epochs  # Number of epochs to use for QAT
BATCH_SIZE = args.batch_size  # Batch size to use for QAT

EXPORT_DIR_OUTPUT = args.export_dir_output

DUMP_COORDS_EMBEDDINGS = args.dump_coords_embeddings

multimask_output = True
model_cfg = args.config
sam2_checkpoint = args.model_checkpoint


class SAM2ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.image_encoder = sam_model.image_encoder
        self.no_mem_embed = sam_model.no_mem_embed

    def forward(self, x: torch.Tensor) -> tuple[Any, Any, Any]:
        backbone_out = self.image_encoder(x)
        # precompute projected level 0 and level 1 features in SAM decoder
        # to avoid running it again on every SAM click
        # print(backbone_out["backbone_fpn"][0].shape) # torch.Size([64, 64, 56, 56])
        # print(backbone_out["backbone_fpn"][1].shape) # torch.Size([64, 64, 28, 28])

        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
            self.model.channel_expansion_0(backbone_out["backbone_fpn"][0])
        )
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
            self.model.channel_expansion_1(backbone_out["backbone_fpn"][1])
        )

        feature_maps = backbone_out["backbone_fpn"][-self.model.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][
            -self.model.num_feature_levels :
        ]

        # print("[DEBUG] vision_pos_embeds", [x.shape for x in vision_pos_embeds])
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        
        # flatten NxCxHxW to HWxNxC
        # Adding 1 dim to make it compliant with IMX500:
        # Cannnot permute batch dim
        feature_maps[0] = feature_maps[0].unsqueeze(1)
        feature_maps[1] = feature_maps[1].unsqueeze(1)
        feature_maps[2] = feature_maps[2].unsqueeze(1)

        # print("[DEBUG] feature_maps", [x.shape for x in feature_maps])
        # [torch.Size([1, 32, 56, 56]), torch.Size([1, 64, 28, 28]), torch.Size([1, 64, 14, 14])]
        vision_feats = [x.flatten(3).permute(0, 3, 1, 2) for x in feature_maps]
        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

        feats = [
            feat.permute(0, 2, 3, 1).reshape(1, 1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]

        # Back in a compatible shape for the model
        feats[0] = feats[0][:, 0]
        feats[1] = feats[1][:, 0]
        feats[2] = feats[2][:, 0]
        
        return feats[0], feats[1], feats[2]


class SAM2ImageDecoder(nn.Module):
    def __init__(self, sam_model: SAM2Base, multimask_output: bool) -> None:
        super().__init__()
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder
        self.model = sam_model
        self.multimask_output = multimask_output
        # Static input for now
        self.dense_embedding = torch.zeros(
            1,
            EMBED_DIM_DECODER,
            sam_model.image_size // sam_model.backbone_stride,
            sam_model.image_size // sam_model.backbone_stride,
            device=device,
        )
        self.sparse_embedding = None

    def forward(
        self,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
        img_size: torch.Tensor,
    ):
        if self.sparse_embedding is None:
            sparse_embedding = self._embed_points(point_coords, point_labels, NUM_POINTS)
        else:
            sparse_embedding = self.sparse_embedding
        # print("[DEBUG] sparse shape", sparse_embedding.shape)
        if DUMP_COORDS_EMBEDDINGS:
            assert LABELS_MODE == "1-click", "Dump embeddings: Only 1-click mode supported for now"
            # Save the sparse embedding
            # verify the file does not exist
            if not os.path.exists("sparse_embedding.npy"):
                to_dump_embedding = self._embed_points(torch.tensor([[[TARGET_POINT_X, TARGET_POINT_Y]]], dtype=torch.float32, device=device), torch.tensor([[1]], dtype=torch.float32, device=device), 1)
                to_dump_embedding = to_dump_embedding.cpu().numpy()
                # print("[DEBUG] to_dump_embedding shape", to_dump_embedding.shape)
                np.save("sparse_embedding.npy", to_dump_embedding)
        # dense_embedding = self._embed_masks(mask_input, has_mask_input)
        # We don't use the mask anyway (could even be removed to model features at some point)
        dense_embedding = self.dense_embedding
        # print("[DEBUG] dense shape", dense_embedding.shape)

        high_res_feats = [high_res_feats_0, high_res_feats_1]
        # print("[DEBUG] image_embed shape", image_embed.shape)
        image_embed = self.model.channel_expansion_2(image_embed)

        masks, iou_predictions, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

        # print("[DEBUG] masks shape", masks.shape)
        # print("[DEBUG] iou_predictions shape", iou_predictions.shape)

        if self.multimask_output:
            # masks = masks[:, 1:, :, :]
            # iou_predictions = iou_predictions[:, 1:]
            masks = torch.stack([masks[:, i, :, :] for i in range(1, 4)], dim=1)
            iou_predictions = torch.stack(
                [iou_predictions[:, i : i + 1] for i in range(1, 4)], dim=1
            )
        else:
            masks, iou_predictions = self.mask_decoder._dynamic_multimask_via_stability(
                masks, iou_predictions
            )

        # masks = torch.clamp(masks, -32.0, 32.0)

        # Do we really need to do this on chip ?
        # masks = F.interpolate(masks, (SAM_INPUT_SIZE, SAM_INPUT_SIZE), mode="bilinear", align_corners=False)

        return masks, iou_predictions

    def _embed_points(
        self, point_coords: torch.Tensor, point_labels: torch.Tensor, num_points: int
    ) -> torch.Tensor:

        # print("[DEBUG] point_coords shape", point_coords.shape)

        point_coords = point_coords + 0.5

        padding_point = torch.zeros(
            (1, num_points, 2), dtype=torch.float32, device=device
        )
        
        padding_label = -torch.ones((1, num_points), device=device)
        point_coords = torch.cat([point_coords, padding_point], dim=1)
        
        # print("[DEBUG] point_labels shape", point_labels.shape)
        # print("[DEBUG] padding_label shape", padding_label.shape)
        
        point_labels = torch.cat([point_labels, padding_label], dim=1)

        # point_coords[:, :, 0] = point_coords[:, :, 0] / self.model.image_size
        # point_coords[:, :, 1] = point_coords[:, :, 1] / self.model.image_size

        point_coords = point_coords / self.model.image_size

        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)
        # print("[DEBUG] point_embedding shape", point_embedding.shape)
        # print("[DEBUG] point_labels.unsqueeze shape", point_labels.unsqueeze(-1).shape)
        point_labels = point_labels.unsqueeze(-1).expand((1, NUM_POINTS*2, 256))

        # We only need this if we use negative prompt points, which we don't
        point_embedding = point_embedding * (point_labels != -1.0)
        point_embedding = (
            point_embedding
            + self.prompt_encoder.not_a_point_embed.weight * (point_labels == -1.0)
        )

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[
                i
            ].weight * (point_labels == i)        

        return point_embedding

    def _embed_masks(
        self, input_mask: torch.Tensor, has_mask_input: torch.Tensor
    ) -> torch.Tensor:
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(
            input_mask
        )
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding


class SAM2DecoderONNXWrapper(nn.Module):
    def __init__(self, sam_model: SAM2Base, multimask_output: bool):
        super().__init__()
        self.decoder = SAM2ImageDecoder(sam_model, multimask_output)
        if STATIC_POINT:
            self.point_coords = torch.tensor(
                [[[TARGET_POINT_X, TARGET_POINT_Y]]], dtype=torch.float, device=device
            )
            self.point_labels = torch.tensor([[1]], dtype=torch.float, device=device)
        mask_input_size = [
            4 * x
            for x in (
                sam_model.image_size // sam_model.backbone_stride,
                sam_model.image_size // sam_model.backbone_stride,
            )
        ]
        self.mask_input = torch.randn(
            1, 1, *mask_input_size, dtype=torch.float, device=device
        )
        self.has_mask_input = torch.tensor([0], dtype=torch.float, device=device)
        self.high_res_feat_0 = torch.randn(1, 32, 32, 32, device=device)
        self.high_res_feat_1 = torch.randn(1, 64, 16, 16, device=device)
        self.orig_im_size = torch.tensor(
            [SAM_INPUT_SIZE, SAM_INPUT_SIZE], dtype=torch.int32, device=device
        )
        # Transform (1, 3, 224, 224) into (1, 64, 8, 8)
        self.transform = torch.nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=28, stride=14, padding=0
        )

    if STATIC_POINT:

        def forward(
            self,
            image_embed: torch.Tensor,
        ):
            masks, iou_predictions = self.decoder(
                self.transform(image_embed),
                self.high_res_feat_0,
                self.high_res_feat_1,
                self.point_coords,
                self.point_labels,
                self.mask_input,
                self.has_mask_input,
                self.orig_im_size,
            )
            return masks, iou_predictions

    else:

        def forward(
            self,
            image_embed: torch.Tensor,
            point_coords: torch.Tensor,
            point_labels: torch.Tensor,
        ):
            masks, iou_predictions = self.decoder(
                self.transform(image_embed),
                self.high_res_feat_0,
                self.high_res_feat_1,
                point_coords,
                point_labels,
                self.mask_input,
                self.has_mask_input,
                self.orig_im_size,
            )
            return masks, iou_predictions


class SAM2ONNXWrapper(nn.Module):
    def __init__(self, sam_model: SAM2Base, multimask_output: bool):
        super().__init__()
        self.input_size = SAM_INPUT_SIZE
        self.encoder = SAM2ImageEncoder(sam_model)
        self.decoder = SAM2ImageDecoder(sam_model, multimask_output)
        if STATIC_POINT:
            self.point_coords = torch.tensor(
                [[[TARGET_POINT_X, TARGET_POINT_Y]]], dtype=torch.float, device=device
            )
            self.point_labels = torch.tensor([[1]], dtype=torch.float, device=device)
        elif STATIC_LABELS:
            if LABELS_MODE == "1-click":
                self.point_labels = torch.tensor([[1]], dtype=torch.float, device=device)
            elif LABELS_MODE == "3-clicks":
                self.point_labels = torch.tensor([[1, 1, 1]], dtype=torch.float, device=device)
            elif LABELS_MODE == "5-clicks":
                self.point_labels = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.float, device=device)
            elif LABELS_MODE == "gt-box":
                self.point_labels = torch.tensor([[2, 3]], dtype=torch.float, device=device)
        mask_input_size = [
            4 * x
            for x in (
                sam2_model.image_size // sam2_model.backbone_stride,
                sam2_model.image_size // sam2_model.backbone_stride,
            )
        ]
        self.mask_input = torch.randn(
            1, 1, *mask_input_size, dtype=torch.float, device=device
        )
        self.has_mask_input = torch.tensor([0], dtype=torch.float, device=device)
        self.orig_im_size = torch.tensor(
            [self.input_size, self.input_size], dtype=torch.int32, device=device
        )
        
    if STATIC_POINT:

        def forward(self, image: torch.Tensor):
            
            # Step 1: Run the encoder
            high_res_feat_0, high_res_feats_1, image_embed = self.encoder(image)

            # Step 2: Run the decoder
            masks, iou_predictions = self.decoder(
                image_embed,
                high_res_feat_0,
                high_res_feats_1,
                self.point_coords,
                self.point_labels,
                self.mask_input,
                self.has_mask_input,
                self.orig_im_size,
            )

            return masks, iou_predictions

    elif STATIC_LABELS:
        
        def forward(self, image: torch.Tensor, point_coords: torch.Tensor):
            # Step 1: Run the encoder
            high_res_feat_0, high_res_feats_1, image_embed = self.encoder(image)

            # Step 2: Run the decoder
            masks, iou_predictions = self.decoder(
                image_embed,
                high_res_feat_0,
                high_res_feats_1,
                point_coords,
                self.point_labels,
                self.mask_input,
                self.has_mask_input,
                self.orig_im_size,
            )

            return masks, iou_predictions

    else:

        def forward(
            self,
            image: torch.Tensor,
            point_coords: torch.Tensor,
            point_labels: torch.Tensor,
        ):
                
            # Step 1: Run the encoder
            high_res_feat_0, high_res_feats_1, image_embed = self.encoder(image)

            # Step 2: Run the decoder
            masks, iou_predictions = self.decoder(
                image_embed,
                high_res_feat_0,
                high_res_feats_1,
                point_coords,
                point_labels,
                self.mask_input,
                self.has_mask_input,
                self.orig_im_size,
            )

            return masks, iou_predictions

register_omegaconf_resolvers()
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

# pritn each layer and size
for name, param in sam2_model.named_parameters():
    print(f"Layer: {name} | Number of parameters: {param.numel()}")

# print the num of params
# Print the total number of parameters in the model
num_params = sum(p.numel() for p in sam2_model.parameters())
print(f"Number of parameters in model: {num_params}")
# Print the total number of parameters in the encoder
num_params = sum(p.numel() for p in sam2_model.image_encoder.parameters())
print(f"Number of trainable parameters in image encoder: {num_params}")
# Print the total number of parameters in the decoder
num_params = sum(p.numel() for p in sam2_model.sam_mask_decoder.parameters())
print(f"Number of trainable parameters in SAM decoder: {num_params}")
# Print the total number of parameters in the prompt encoder
num_params = sum(p.numel() for p in sam2_model.sam_prompt_encoder.parameters())
print(f"Number of trainable parameters in SAM prompt encoder: {num_params}")
# Print the total number of parameters in the memory encoder
if (
    hasattr(sam2_model, "memory_encoder")
    and sam2_model.memory_encoder is not None
):
    num_params = sum(p.numel() for p in sam2_model.memory_encoder.parameters())
    print(f"Number of trainable parameters in memory encoder: {num_params}")
else:
    print("Model does not have memory encoder")
# Print the total number of parameters in the memory attention
if (
    hasattr(sam2_model, "memory_attention")
    and sam2_model.memory_attention is not None
):
    num_params = sum(
        p.numel() for p in sam2_model.memory_attention.parameters()
    )
    print(f"Number of trainable parameters in memory attention: {num_params}")
else:
    print("Model does not have memory attention")


input_size = SAM_INPUT_SIZE
img = torch.randn(1, 3, input_size, input_size, device=device)
if not STATIC_POINT:
    if not STATIC_LABELS:
        point_coords = torch.tensor(
            [[[TARGET_POINT_X, TARGET_POINT_Y]]], dtype=torch.float, device=device
        )
        point_labels = torch.tensor([[1]], dtype=torch.float, device=device)
        num_points = 1
    else:
        if LABELS_MODE == "1-click":
            point_coords = torch.tensor(
                [[[TARGET_POINT_X, TARGET_POINT_Y]]], dtype=torch.float, device=device
            )
        elif LABELS_MODE == "3-clicks":
            point_coords = torch.tensor(
                [[
                    [TARGET_POINT_X, TARGET_POINT_Y],
                    [TARGET_POINT_X + 10, TARGET_POINT_Y + 10],
                    [TARGET_POINT_X + 20, TARGET_POINT_Y + 20],
                ]],
                dtype=torch.float,
                device=device,
            )
        elif LABELS_MODE == "5-clicks":
            point_coords = torch.tensor(
                [[
                    [TARGET_POINT_X, TARGET_POINT_Y],
                    [TARGET_POINT_X + 10, TARGET_POINT_Y + 10],
                    [TARGET_POINT_X + 20, TARGET_POINT_Y + 20],
                    [TARGET_POINT_X + 30, TARGET_POINT_Y + 30],
                    [TARGET_POINT_X + 40, TARGET_POINT_Y + 40],
                ]],
                dtype=torch.float,
                device=device,
            )
        elif LABELS_MODE == "gt-box":
            point_coords = torch.tensor(
                [[
                    [TARGET_POINT_X, TARGET_POINT_Y],
                    [TARGET_POINT_X + 10, TARGET_POINT_Y + 10]
                ]],
                dtype=torch.float,
                device=device,
            )

if "encoder" in EXPORTED_MODELS and EXPORT_FP:
    sam2_encoder = SAM2ImageEncoder(sam2_model)
    high_res_feat_0, high_res_feats_1, image_embed = sam2_encoder(img)

    torch.onnx.export(
        sam2_encoder,
        img,
        f"sam2_encoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["image_embed"],
    )

if "decoder" in EXPORTED_MODELS and EXPORT_FP:
    sam2_decoder = SAM2ImageDecoder(sam2_model, multimask_output=multimask_output)

    embed_dim = sam2_model.sam_prompt_encoder.embed_dim
    embed_size = (
        sam2_model.image_size // sam2_model.backbone_stride,
        sam2_model.image_size // sam2_model.backbone_stride,
    )
    mask_input_size = [4 * x for x in embed_size]
    print(embed_dim, embed_size, mask_input_size)

    input_size = SAM_INPUT_SIZE
    point_coords = torch.randint(
        low=0, high=input_size, size=(1, 3, 2), dtype=torch.float, device=device
    )
    point_labels = torch.tensor([[1, 1]], dtype=torch.float, device=device)
    mask_input = torch.randn(1, 1, *mask_input_size, dtype=torch.float, device=device)
    has_mask_input = torch.tensor([0], dtype=torch.float, device=device)
    orig_im_size = torch.tensor(
        [input_size, input_size], dtype=torch.int32, device=device
    )

    masks, scores = sam2_decoder(
        image_embed,
        high_res_feat_0,
        high_res_feats_1,
        point_coords,
        point_labels,
        mask_input,
        has_mask_input,
        orig_im_size,
    )

    torch.onnx.export(
        sam2_decoder,
        (
            image_embed,
            high_res_feat_0,
            high_res_feats_1,
            point_coords,
            point_labels,
            mask_input,
            has_mask_input,
            orig_im_size,
        ),
        "sam2_decoder.onnx",
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=[
            "image_embed",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
            "orig_im_size",
        ],
        output_names=["masks", "iou_predictions"],
        dynamic_axes={
            "point_coords": {0: "num_labels", 1: "num_points"},
            "point_labels": {0: "num_labels", 1: "num_points"},
            "mask_input": {0: "num_labels"},
            "has_mask_input": {0: "num_labels"},
        },
    )

if "full" in EXPORTED_MODELS and EXPORT_FP:
    sam2_wrapper = SAM2ONNXWrapper(sam2_model, multimask_output=multimask_output)

    if STATIC_POINT:
        torch.onnx.export(
            sam2_wrapper,
            (img),
            NAME_FP_STATIC_PT,
            export_params=True,
            opset_version=16,  # Ensure compatibility with IMX500's ONNX runtime
            do_constant_folding=True,
            input_names=["image"],
            output_names=["masks", "iou_predictions"],
            dynamic_axes={
                "image": {0: "batch_size"},
            },
            verbose=True,
            keep_initializers_as_inputs=False,
        )
        
        def get_dummy_input(device="cuda", input_size=128):
            dummy_image = img.to(device)
            return dummy_image
        
        def compute_macs_and_params(model):
            model.eval()
            dummy_input = get_dummy_input(device)

            with torch.no_grad():
                flops = FlopCountAnalysis(model, dummy_input)
                params = parameter_count(model)

            print("FLOPs (MACs): {:.2f} G".format(flops.total() / 1e9))
            print("Number of Parameters: {:.2f} M".format(params[''] / 1e6))
        
        compute_macs_and_params(sam2_wrapper)
        
    elif STATIC_LABELS:
        torch.onnx.export(
            sam2_wrapper,
            (img, point_coords),
            NAME_FP_STATIC_LABELS,
            export_params=True,
            opset_version=16,  # Ensure compatibility with IMX500's ONNX runtime
            do_constant_folding=True,
            input_names=["image", "point_coords"],
            output_names=["masks", "iou_predictions"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "point_coords": {0: "num_labels", 1: "num_points"},
            },
            verbose=True,
            keep_initializers_as_inputs=False,
        )
    else:
        torch.onnx.export(
            sam2_wrapper,
            (img, point_coords, point_labels),
            NAME_FP,
            export_params=True,
            opset_version=16,  # Ensure compatibility with IMX500's ONNX runtime
            do_constant_folding=True,
            input_names=["image", "point_coords", "point_labels"],
            output_names=["masks", "iou_predictions"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "point_coords": {0: "num_labels", 1: "num_points"},
                "point_labels": {0: "num_labels", 1: "num_points"},
            },
            verbose=True,
            keep_initializers_as_inputs=False,
        )


transform = transforms.Compose(
    [
        transforms.Resize((SAM_INPUT_SIZE, SAM_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if STATIC_POINT:

    def representative_data_gen():
        for file in os.listdir(IMG_DIR)[:NUM_SAMPLES]:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(IMG_DIR, file)
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)
                #print("rep data shape", img_tensor.shape)
                yield [img_tensor]

elif STATIC_LABELS:
    def representative_data_gen():
        for file in os.listdir(IMG_DIR)[:NUM_SAMPLES]:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(IMG_DIR, file)
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)
                if LABELS_MODE == "1-click":
                    point_coords = torch.randint(
                        low=0, high=SAM_INPUT_SIZE, size=(1, 1, 2), dtype=torch.float
                    )
                elif LABELS_MODE == "3-clicks":
                    point_coords = torch.randint(
                        low=0,
                        high=SAM_INPUT_SIZE,
                        size=(1, 3, 2),
                        dtype=torch.float,
                    )
                elif LABELS_MODE == "5-clicks":
                    point_coords = torch.randint(
                        low=0,
                        high=SAM_INPUT_SIZE,
                        size=(1, 5, 2),
                        dtype=torch.float,
                    )
                elif LABELS_MODE == "gt-box":
                    point_coords = torch.randint(
                        low=0,
                        high=SAM_INPUT_SIZE,
                        size=(1, 2, 2),
                        dtype=torch.float,
                    )
                    
                print("rep data shape", img_tensor.shape)
                yield [img_tensor, point_coords]

else:

    def representative_data_gen():
        for file in os.listdir(IMG_DIR)[:NUM_SAMPLES]:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(IMG_DIR, file)
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0)
                # point_coords = torch.tensor(
                #     [[[TARGET_POINT_X, TARGET_POINT_Y]]], dtype=torch.float
                # print )
                point_coords = torch.randint(
                    low=0, high=SAM_INPUT_SIZE, size=(1, 1, 2), dtype=torch.float
                )
                point_labels = torch.tensor([[1]], dtype=torch.float)
                print("rep data shape", img_tensor.shape)
                yield [img_tensor, point_coords, point_labels]


# mct.exporter.pytorch_export_model(sam2_wrapper, save_model_path='sam2_full.onnx', repr_dataset=representative_data_gen)

tpc = mct.get_target_platform_capabilities(
    "pytorch", "imx500", target_platform_version="v1"
)

if "encoder" in EXPORTED_MODELS and EXPORT_QUANT:
    if QUANT_METHOD == "PTQ":
        quantized_model, quantization_infos = (
            mct.ptq.pytorch_post_training_quantization(
                sam2_encoder, representative_data_gen, target_platform_capabilities=tpc
            )
        )
        mct.exporter.pytorch_export_model(
            quantized_model,
            save_model_path="q-encoder.onnx",
            repr_dataset=representative_data_gen,
            onnx_opset_version=20,
        )
    else:
        raise NotImplementedError(
            "QAT is not implemented yet for encoder. Please use PTQ for now."
        )

if "decoder" in EXPORTED_MODELS and EXPORT_QUANT:
    sam2_decoderw = SAM2DecoderONNXWrapper(
        sam2_model, multimask_output=multimask_output
    )
    if QUANT_METHOD == "PTQ":
        quantized_model, quantization_infos = (
            mct.ptq.pytorch_post_training_quantization(
                sam2_decoderw, representative_data_gen, target_platform_capabilities=tpc
            )
        )
        mct.exporter.pytorch_export_model(
            quantized_model,
            save_model_path="q-decoder.onnx",
            repr_dataset=representative_data_gen,
            onnx_opset_version=20,
        )
    else:
        raise NotImplementedError(
            "QAT is not implemented yet for decoder. Please use PTQ for now."
        )

if "full" in EXPORTED_MODELS and EXPORT_QUANT:
    if "sam2_wrapper" not in locals():
        sam2_wrapper = SAM2ONNXWrapper(sam2_model, multimask_output=multimask_output)

    if QUANT_METHOD == "PTQ":
        
            if LOAD_CKPT_BEFORE_PTQ:
                # Load the checkpoint
                sam2_wrapper.load_state_dict(torch.load(QAT_CKPT), strict=False)
        
            quantized_model, quantization_infos = (
                mct.ptq.pytorch_post_training_quantization(
                    sam2_wrapper, representative_data_gen, target_platform_capabilities=tpc
                )
            )
            
            if LOAD_CKPT_BEFORE_PTQ:
                if STATIC_POINT:
                    name = "qat-ultra-tiny-sam-static.onnx"
                else:
                    name = NAME_QAT
            else:
                if STATIC_POINT:
                    name = "ptq-ultra-tiny-sam-static.onnx"
                else:
                    name = NAME_PTQ
            mct.exporter.pytorch_export_model(
                quantized_model,
                save_model_path=name,
                repr_dataset=representative_data_gen,
                onnx_opset_version=20,
            )
    
    elif QUANT_METHOD == "GPTQ":
        raise NotImplementedError(
            "GPTQ is not implemented yet for full model. Please use PTQ for now."
        )
        
        gptq_config = mct.gptq.get_pytorch_gptq_config(n_epochs=5)
        gptq_config.hessian_weights_config.hessian_batch_size = 1
        
        tpc = mct.get_target_platform_capabilities(
            "pytorch", "imx500", target_platform_version="v1"
        )
        
        quantized_model, quantization_infos = (
            mct.gptq.pytorch_gradient_post_training_quantization(
                sam2_wrapper,
                representative_data_gen,
                gptq_config=gptq_config,
                target_platform_capabilities=tpc,
            )
        )
        
        mct.exporter.pytorch_export_model(
                quantized_model,
                save_model_path="gptq-ultra-tiny-sam.onnx",
                repr_dataset=representative_data_gen,
                onnx_opset_version=20,
            )
            
    else: # QAT
        assert STATIC_POINT == False, "QAT is not supported for static point mode"
        # create the export directory
        os.makedirs(EXPORT_DIR_OUTPUT, exist_ok=True)
        
        # The MCT converts a floating-point model into a quantized model using post-training quantization.
        # The returned model includes trainable quantizers and is ready for fine-tuning, making it a "QAT-ready" model.
        # qat_model, quantization_infos = (
        #     mct.qat.pytorch_quantization_aware_training_init_experimental(
        #         sam2_wrapper,
        #         representative_data_gen,
        #         target_platform_capabilities=tpc,
        #     )
        # )
        import torch.quantization
        from torch.quantization import (convert, get_default_qat_qconfig,
                                        prepare_qat)

        model_fp32 = sam2_wrapper.train()

        # Set symmetric per-channel quantization config (for conv weights)
        qconfig = torch.quantization.QConfig(
            activation=torch.quantization.FakeQuantize.with_args(observer=torch.quantization.MovingAverageMinMaxObserver, qscheme=torch.per_tensor_symmetric),
            weight=torch.quantization.FakeQuantize.with_args(observer=torch.quantization.PerChannelMinMaxObserver, qscheme=torch.per_channel_symmetric)
        )

        model_fp32.qconfig = qconfig
        
        qat_model = prepare_qat(model_fp32, inplace=False)
        
        device_qat = 'cuda'
        qat_model = qat_model.to(device_qat)
        
        ### Here we need to trai    n the model with QAT
        ### I think at this point only a basic training method is required

        # First the datloaders
        # === IMPORTS ===
        from functools import partial

        import cv2
        import numpy as np
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from training.dataset.sam2_datasets import TorchTrainMixedDataset
        from training.dataset.transforms import (ColorJitter, ComposeAPI,
                                                 NormalizeAPI, RandomGrayscale,
                                                 RandomHorizontalFlip,
                                                 RandomResizeAPI, ToTensorAPI)
        from training.dataset.vos_dataset import VOSDataset
        from training.dataset.vos_raw_dataset import SA1BRawDataset
        from training.dataset.vos_sampler import RandomUniformSampler
        from training.utils.data_utils import collate_fn

        # === SETUP ===
        # === CONFIGURATION VALUES (replace with actual values or config access) ===
        train_img_folder = None
        train_gt_folder = None
        val_img_folder = None
        val_gt_folder = None
        file_list_train = "/datasets/magirard/SA1B_preprocessed_up_150/file_list_train.txt"
        file_list_val = "/datasets/magirard/SA1B_preprocessed_up_150/file_list_val.txt"

        resolution = 128  # example for ${scratch.resolution}
        max_num_objects = 3
        multiplier = 2
        train_batch_size = 1
        val_batch_size = 1
        phases_per_epoch = 1
        num_train_workers = 12
        num_val_workers = 6
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        lr = 1e-6 # Should be set to the one at the end of fp training

        # === TRAIN TRANSFORMS ===
        train_transforms = ComposeAPI(
            transforms=[
                RandomHorizontalFlip(consistent_transform=True),
                # RandomAffine intentionally skipped (commented out in YAML)
                RandomResizeAPI(
                    sizes=resolution, square=True, consistent_transform=True
                ),
                ColorJitter(
                    consistent_transform=True,
                    brightness=0.1,
                    contrast=0.03,
                    saturation=0.03,
                    hue=None,
                ),
                RandomGrayscale(p=0.05, consistent_transform=True),
                ColorJitter(
                    consistent_transform=False,
                    brightness=0.1,
                    contrast=0.05,
                    saturation=0.05,
                    hue=None,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=mean, std=std),
            ]
        )

        # === VAL TRANSFORMS ===
        val_transforms = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=resolution, square=True, consistent_transform=True
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=mean, std=std),
            ]
        )

        # === VIDEO DATASETS ===
        video_dataset_train = SA1BRawDataset(
            img_folder=train_img_folder, gt_folder=train_gt_folder, file_list_txt=file_list_train
        )

        video_dataset_val = SA1BRawDataset(
            img_folder=val_img_folder, gt_folder=val_gt_folder, file_list_txt=file_list_val
        )

        # === SAMPLERS ===
        sampler_train = RandomUniformSampler(
            num_frames=1, max_num_objects=max_num_objects
        )

        sampler_val = RandomUniformSampler(
            num_frames=1, max_num_objects=max_num_objects
        )

        # === VOS DATASETS ===
        vos_dataset_train = VOSDataset(
            training=True,
            video_dataset=video_dataset_train,
            sampler=sampler_train,
            transforms=[train_transforms],
            multiplier=multiplier,
        )

        vos_dataset_val = VOSDataset(
            training=True,  # same as train, your config also uses `training: true` for val
            video_dataset=video_dataset_val,
            sampler=sampler_val,
            transforms=[val_transforms],
            multiplier=multiplier,
        )

        # === COLLATE FUNCTION ===
        collate_fn_partial = partial(collate_fn, dict_key="all")

        # === TRAIN DATALOADER ===
        train_dataloader = DataLoader(
            vos_dataset_train,
            batch_size=train_batch_size,  # define this elsewhere
            shuffle=True,  # shuffle for training
            num_workers=num_val_workers,  # define this elsewhere (e.g., 4 or 8)
            pin_memory=True,  # speeds up host-to-device copies
            drop_last=False,  # keep final incomplete batch
            collate_fn=collate_fn_partial,  # your provided collate function
        )

        # === VAL DATALOADER ===
        val_dataloader = DataLoader(
            vos_dataset_val,
            batch_size=val_batch_size,  # usually same as train, or larger if val fits in memory
            shuffle=False,  # no shuffle for validation
            num_workers=num_train_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_partial,
        )

        # Need to adapt this for gt boxes
        def pick_random_points_in_mask(masks, num_points, rng=None):
            """
            Args:
                masks: torch.Tensor of shape [B, T, H, W] (e.g., [4, 3, 128, 128])
                num_points: int, number of points to sample per batch item
                rng: optional, random generator (torch.Generator)

            Returns:
                point_coords: shape [B, num_points, 2] → (x, y) normalized to [0, 1]
                point_labels: shape [B, num_points] → all ones
                t_list: list of selected frame indices per batch item (length B)
            """
            
            if rng is None:
                rng = torch.Generator()
                rng.manual_seed(torch.seed())

            B, T, H, W = masks.shape
            point_coords_list = []
            point_labels_list = []
            t_list = []

            for b in range(B):
                # Pick a random frame per batch item
                t = torch.randint(0, T, (1,), generator=rng).item()
                t_list.append(t)
                mask = masks[b, t]  # shape [H, W]

                # Get (y, x) indices where mask == 1
                ys, xs = torch.nonzero(mask, as_tuple=True)
                num_mask_pixels = ys.shape[0]

                if num_mask_pixels == 0:
                    raise ValueError(f"No foreground pixels found in mask for batch item {b}.")

                # If fewer mask pixels than requested points, sample with replacement
                replace = num_points > num_mask_pixels
                indices = torch.randint(0, num_mask_pixels, (num_points,), generator=rng)

                selected_ys = ys[indices]
                selected_xs = xs[indices]

                # Normalize to [0, 1] (x, y) order
                coords = torch.stack(
                    [selected_xs, selected_ys], dim=-1
                )  # shape [num_points, 2]

                point_coords_list.append(coords.unsqueeze(0))  # [1, num_points, 2]

                # Create point_labels = 1
                point_labels = torch.ones((1, num_points), dtype=torch.float32)
                point_labels_list.append(point_labels)

            # Stack over batch dimension
            point_coords = torch.cat(point_coords_list, dim=0)  # [B, num_points, 2]
            point_labels = torch.cat(point_labels_list, dim=0)  # [B, num_points, 1]
            return point_coords, point_labels, t_list

        # Print one iteration of the dataloader
        for i, data in enumerate(train_dataloader):
            print(f"Batch {i}:")
            print(data.keys())
            print(f"  Image shape: {data.img_batch.shape}")
            print(f"  Mask shape: {data.masks.shape}")
            print(f"  Metadata: {data.metadata.shape}")
            # pick random points in the mask
            point_coords, point_labels, t_list = pick_random_points_in_mask(
                data.masks, num_points=NUM_POINTS
            )
            print(f"Random point coords: {point_coords}")
            print(f"Random point labels: {point_labels}")
            break

        # === TRAINING LOOP ===
        # Use a loss modified from fns_losses
            
        # === CUSTOM LOSS ===
        from collections import defaultdict

        from training.trainer import CORE_LOSS_KEY
        from training.utils.distributed import (get_world_size,
                                                is_dist_avail_and_initialized)

        def dice_loss(inputs, targets, num_objects):
            inputs = inputs.sigmoid()
            B, N, H, W = inputs.shape  # B=batch, N=num_masks
            inputs = inputs.flatten(2)  # [B, N, HW]
            targets_expanded = targets.expand(B, N, H, W)  # [B, N, H, W]
            targets = targets_expanded.flatten(2)  # [B, N, HW]            numerator = 2 * (inputs * targets).sum(-1)  # [B, N]
            numerator = 2 * (inputs * targets).sum(-1)  # [B, N]
            denominator = inputs.sum(-1) + targets.sum(-1)  # [B, N]
            loss = 1 - (numerator + 1) / (denominator + 1)  # [B, N]
            return loss  # don't sum here — let outer logic pick best mask per sample


        def sigmoid_focal_loss(inputs, targets, num_objects, alpha=0.25, gamma=2, loss_on_multimask=False):
            prob = inputs.sigmoid()
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            p_t = prob * targets + (1 - prob) * (1 - targets)
            loss = ce_loss * ((1 - p_t) ** gamma)
            if alpha >= 0:
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
                loss = alpha_t * loss
            if loss_on_multimask:
                assert loss.dim() == 4  # [B, num_masks, H, W]
                return loss.flatten(2).mean(-1) / num_objects  # [B, num_masks]
            return loss.mean(1).sum() / num_objects


        def iou_loss(inputs, targets, pred_ious, num_objects, use_l1_loss=False):
            assert inputs.dim() == 4 and targets.dim() == 4
            pred_mask = inputs.flatten(2) > 0
            gt_mask = targets.flatten(2) > 0
            area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
            area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
            actual_ious = area_i / torch.clamp(area_u, min=1.0)
            if use_l1_loss:
                loss = F.l1_loss(pred_ious.squeeze(-1), actual_ious, reduction="none")
            else:
                loss = F.mse_loss(pred_ious.squeeze(-1), actual_ious, reduction="none")
            return loss.sum() / num_objects


        class MultiMasksAndIousLoss(nn.Module):
            def __init__(
                self,
                weight_dict,
                focal_alpha=0.25,
                focal_gamma=2,
                iou_use_l1_loss=False,
            ):
                super().__init__()
                self.weight_dict = weight_dict
                self.focal_alpha = focal_alpha
                self.focal_gamma = focal_gamma
                self.iou_use_l1_loss = iou_use_l1_loss

                assert "loss_mask" in self.weight_dict
                assert "loss_dice" in self.weight_dict
                assert "loss_iou" in self.weight_dict

            def forward(
                self,
                masks_pred,   # list of [B, num_masks, H, W]
                ious_pred,    # list of [B, num_masks, 1]
                targets_batch       # [B, 1, H, W]
            ):
                assert len(masks_pred) == len(ious_pred)
                num_objects = torch.tensor(
                    (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
                )
                if is_dist_avail_and_initialized():
                    torch.distributed.all_reduce(num_objects)
                num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

                losses = defaultdict(float)
                for pred_masks, pred_ious in zip(masks_pred, ious_pred):
                    cur_losses = self._compute_losses(pred_masks, pred_ious, targets_batch, num_objects)
                    for k, v in cur_losses.items():
                        losses[k] += v

                losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
                return losses

            def _compute_losses(self, pred_masks, pred_ious, target_masks, num_objects):
                target_masks = target_masks.expand_as(pred_masks).float()

                loss_multimask = sigmoid_focal_loss(
                    pred_masks, target_masks, num_objects, alpha=self.focal_alpha, gamma=self.focal_gamma, loss_on_multimask=True
                )
                loss_multidice = dice_loss(pred_masks, target_masks, num_objects)
                loss_iou = iou_loss(pred_masks, target_masks, pred_ious, num_objects, use_l1_loss=True)  # ADD THIS LINE
                
                if pred_masks.size(1) > 1:
                    loss_combo = loss_multimask * self.weight_dict["loss_mask"] + loss_multidice * self.weight_dict["loss_dice"]
                    best_loss_inds = torch.argmin(loss_combo, dim=-1)
                    batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
                    loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
                    loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
                    loss_iou = loss_iou.unsqueeze(0)
                else:
                    loss_mask = loss_multimask
                    loss_dice = loss_multidice
                    loss_iou = loss_iou

                losses = {
                    "loss_mask": loss_mask,
                    "loss_dice": loss_dice,
                    "loss_iou": loss_iou,
                }

                return losses

            def reduce_loss(self, losses):
                reduced_loss = 0.0
                for loss_key, weight in self.weight_dict.items():
                    if loss_key not in losses:
                        raise ValueError(f"{type(self)} doesn't compute {loss_key}")
                    if weight != 0:
                        reduced_loss += losses[loss_key] * weight
                return reduced_loss


        # Instantiate the class
        weight_dict = {
            "loss_mask": 2,
            "loss_dice": 1,
            "loss_iou": 0,
        }
        
        loss_fn = MultiMasksAndIousLoss(
            weight_dict=weight_dict,
            iou_use_l1_loss=True,
            focal_gamma=0.0,
            focal_alpha=-1.0,
        )

        # Use the same optimizer
        from fvcore.common.param_scheduler import (ConstantParamScheduler,
                                                   CosineParamScheduler)
        from torch.optim import AdamW
        from training.optimizer import (GradientClipper,
                                        _unix_pattern_to_parameter_names,
                                        get_module_cls_to_param_names,
                                        layer_decay_param_modifier,
                                        map_scheduler_cfgs_to_param_groups,
                                        validate_param_group_params)

        param_allowlist = {name for name, _ in qat_model.named_parameters()}
        named_parameters = {
            name: param
            for name, param in qat_model.named_parameters()
            if name in param_allowlist
        }

        # === Mimic options_conf ===
        all_parameter_names = set(named_parameters.keys())
        module_cls_to_all_param_names = get_module_cls_to_param_names(
            qat_model, param_allowlist
        )

        # Scheduler configs (mimicking hydra.utils.instantiate)
        scheduler_cfgs_per_option = {
            "lr": [
                {
                    "scheduler": CosineParamScheduler(
                        start_value=lr, end_value=lr / 10
                    ),
                    "param_names": set(all_parameter_names),
                },
                # In the original training loop we discriminate the vision training,
                # but here lost because of the wrapper
            ],
            "weight_decay": [
                {
                    "scheduler": ConstantParamScheduler(0.0),
                    "param_names": set(all_parameter_names),
                },
                # To look later
                # {
                #     "scheduler": ConstantParamScheduler(0.0),
                #     "param_names": _unix_pattern_to_parameter_names(
                #         {
                #             "param_names": ["*bias*"],
                #             "module_cls_names": ["torch.nn.LayerNorm"],
                #         },
                #         all_parameter_names,
                #         module_cls_to_all_param_names,
                #     ),
                # },
            ],
        }

        def set_default_parameters(scheduler_cfgs, all_parameter_names):

            constraints = [
                scheduler_cfg['parameter_names']
                for scheduler_cfg in scheduler_cfgs
                if scheduler_cfg['parameter_names'] is not None
            ]
            if len(constraints) == 0:
                default_params = set(all_parameter_names)
            else:
                default_params = all_parameter_names - set.union(*constraints)
            default_count = 0
            for scheduler_cfg in scheduler_cfgs:
                if scheduler_cfg['parameter_names'] is None:
                    scheduler_cfg['parameter_names'] = default_params
                    default_count += 1
            assert default_count <= 1, "Only one scheduler per option can be default"
            if default_count == 0:
                # No default scheduler specified, add a default, but without any scheduler
                # for that option
                scheduler_cfgs.append({"parameter_names": default_params})

        # Wrap into the scheduler config list
        all_scheduler_cfgs = []
        for option, scheduler_cfgs in scheduler_cfgs_per_option.items():
            for config in scheduler_cfgs:
                config["option"] = option
                config["parameter_names"] = config["param_names"]
                del config["param_names"]
            print(f"Scheduler configs keys: {[x.keys() for x in scheduler_cfgs]}")
            set_default_parameters(scheduler_cfgs, all_parameter_names)
            all_scheduler_cfgs.append(scheduler_cfgs)

        # === Apply param group modifiers (layer decay) ===
        # This was in original code but again we lost track of the parameters name when wrapping
        # all_scheduler_cfgs = layer_decay_param_modifier(
        #     scheduler_cfgs=all_scheduler_cfgs,
        #     model=qat_model,
        #     layer_decay_value=0.9,
        #     apply_to="image_encoder.trunk",
        #     overrides=[{"pattern": "*pos_embed*", "value": 1.0}],
        # )

        # === Map scheduler configs to param groups ===
        schedulers, param_groups = map_scheduler_cfgs_to_param_groups(
            all_scheduler_cfgs, named_parameters
        )

        # === (Optional) Validate param groups ===
        validate_param_group_params(param_groups, qat_model)

        # === Instantiate optimizer ===
        optimizer = AdamW(param_groups,
            lr=lr,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            eps=1e-10,
            amsgrad=False,
            foreach=None,
            capturable=False,
            differentiable=False,
            maximize=False,
            fused=None,
        )

        # === Setup gradient clipper ===
        gradient_clipper = GradientClipper(max_norm=0.1, norm_type=2)

        # === Training loop ===
        # batch size of 1 (mandatory with our ONNX wrapper)
        # do just few epochs, 1 point per mask
        # low lr
        VIRTUAL_BATCH_SIZE = BATCH_SIZE
        GRAD_ACCUM_STEPS = VIRTUAL_BATCH_SIZE  # since physical batch_size == 1
                                
        for epoch in range(NUM_EPOCHS):  # e.g. num_epochs = 1
            num_virtual_batches = (len(train_dataloader) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS

            with tqdm(total=num_virtual_batches, desc=f"Epoch {epoch+1}") as pbar:
                running_loss_sum = 0.0
                running_loss_count = 0

                optimizer.zero_grad()
                local_step = 0  # counts samples

                for i, data in enumerate(train_dataloader):
                    try:
                        point_coords, point_labels, t_list = pick_random_points_in_mask(
                            data.masks.view(train_batch_size, 3, data.masks.size(-2), data.masks.size(-1)),
                            num_points=NUM_POINTS
                        )
                    except RuntimeError as e:
                        #print(f"Error in pick_random_points_in_mask: {e}")
                        continue

                    img_batch = data.img_batch.to(device_qat).squeeze(0)
                    point_coords = point_coords.to(device_qat)
                    point_labels = point_labels.to(device_qat)

                    preds, iou_predictions = qat_model(
                        img_batch,
                        point_coords,
                        #point_labels,
                    )
                    preds = F.interpolate(preds, size=(128, 128),
                                        mode='bilinear', align_corners=False)

                    B = data.masks.shape[0]
                    selected_masks = []
                    for b in range(B):
                        t_b = t_list[b]
                        selected_masks.append(
                            data.masks[b:b+1, t_b:t_b+1, :, :]
                        )
                    selected_mask = torch.cat(selected_masks, dim=0)

                    loss_dict = loss_fn(
                        preds.unsqueeze(0),
                        iou_predictions.unsqueeze(0),
                        selected_mask.to(device_qat),
                    )
                    batch_loss = loss_dict[CORE_LOSS_KEY]
                    loss_scaled = batch_loss / GRAD_ACCUM_STEPS
                    loss_scaled.backward()                    

                    running_loss_sum += batch_loss.item()
                    running_loss_count += 1
                    local_step += 1

                    # Only step when we’ve accumulated enough
                    if local_step % GRAD_ACCUM_STEPS == 0:
                        gradient_clipper(qat_model) # => not used atm
                        optimizer.step()
                        optimizer.zero_grad()

                        mean_loss = running_loss_sum / running_loss_count
                        pbar.set_postfix({
                            "batch_loss": f"{batch_loss.item():.4f}",
                            "mean_loss":  f"{mean_loss:.4f}"
                        })
                        pbar.update(1)  # <- advance by one *virtual* batch
                    
                    
                    if running_loss_count % VIRTUAL_BATCH_SIZE == 0:
                        for j in range(img_batch.size(0)):
                            img = img_batch[j]
                            # Convert masks to numpy arrays (GT mask and Pred mask)
                            # print(f"GT mask shape: {selected_mask.shape}")
                            # print(f"Pred mask shape: {preds.shape}")
                            gt_mask = selected_mask[0, 0].cpu().numpy()  # shape [128, 128]
                            best_iou_index = torch.argmax(iou_predictions[j, :, 0]).item()  # Get the index of the best IoU prediction
                            # print best iou index, iou and min max for the pred mask
                            # print(f"Best IoU index: {best_iou_index}")
                            # print(f"Pred mask min: {preds[j, best_iou_index].min()}, max: {preds[j, best_iou_index].max()}")
                            pred_mask = (preds[j, best_iou_index].detach().cpu().numpy() > 0).astype(np.uint8)  # binary mask

                            # Convert img_batch to numpy image (assuming img_batch is [C, H, W])
                            img = img.permute(1, 2, 0).cpu().numpy()
                            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Unnormalize
                            img = np.clip(img * 255, 0, 255).astype(np.uint8)
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

                            # Visualization: GT and Pred masks overlaid on image
                            img_with_gt_mask = img.copy()
                            img_with_pred_mask = img.copy()

                            # Color for the masks (red for GT and green for prediction)
                            mask_color_gt = (255, 0, 0)  # Red for GT
                            mask_color_pred = (0, 255, 0)  # Green for Pred

                            # Apply ground truth mask to image
                            img_with_gt_mask[:, :, 0] = np.where(gt_mask > 0, mask_color_gt[0], img_with_gt_mask[:, :, 0])
                            img_with_gt_mask[:, :, 1] = np.where(gt_mask > 0, mask_color_gt[1], img_with_gt_mask[:, :, 1])
                            img_with_gt_mask[:, :, 2] = np.where(gt_mask > 0, mask_color_gt[2], img_with_gt_mask[:, :, 2])

                            # Apply predicted mask to image
                            img_with_pred_mask[:, :, 0] = np.where(pred_mask > 0, mask_color_pred[0], img_with_pred_mask[:, :, 0])
                            img_with_pred_mask[:, :, 1] = np.where(pred_mask > 0, mask_color_pred[1], img_with_pred_mask[:, :, 1])
                            img_with_pred_mask[:, :, 2] = np.where(pred_mask > 0, mask_color_pred[2], img_with_pred_mask[:, :, 2])

                            for point in point_coords.cpu().numpy():
                                x, y = point[0]  # Point in the (x, y) format
                                
                                # Draw a marker for the point (e.g., a star marker in yellow)
                                cv2.drawMarker(
                                    img_with_gt_mask, (int(x), int(y)), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=8, thickness=1
                                )
                                cv2.drawMarker(
                                    img_with_pred_mask, (int(x), int(y)), (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=8, thickness=1
                                )

                            # Save the images with masks
                            save_dir = "visualizations-qat"  # Change to your desired directory
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)

                            # Save the images with masks overlaid
                            cv2.imwrite(f"{save_dir}/img_{epoch}_{running_loss_count//VIRTUAL_BATCH_SIZE}_with_gt_mask.png", img_with_gt_mask)
                            cv2.imwrite(f"{save_dir}/img_{epoch}_{running_loss_count//VIRTUAL_BATCH_SIZE}_with_pred_mask.png", img_with_pred_mask)
                            
                        if i > MAX_BATCHES * VIRTUAL_BATCH_SIZE:
                            break
                            
                # Handle leftovers at epoch end
                remainder = running_loss_count % GRAD_ACCUM_STEPS
                if remainder != 0:
                    gradient_clipper(qat_model)
                    optimizer.step()
                    optimizer.zero_grad()
                    mean_loss = running_loss_sum / running_loss_count
                    pbar.set_postfix({
                        "batch_loss": f"{batch_loss.item():.4f}",
                        "mean_loss":  f"{mean_loss:.4f}"
                    })
                    pbar.update(1)  # <- final leftover virtual batch
        
            # === Save the model ===
            torch.save(
                qat_model.state_dict(),
                f"{EXPORT_DIR_OUTPUT}/ultra-tiny-sam-qat-{epoch}.pth",
            )
            
            print(f"Epoch {epoch+1} completed. Model saved as 'ultra-tiny-sam-qat-{epoch}.pt'.")                                    