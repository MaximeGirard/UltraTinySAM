import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List

from training.trainer import CORE_LOSS_KEY
from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(inputs, targets, num_objects, alpha=0.25, gamma=2, loss_on_multimask=False):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if loss_on_multimask:
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects
    return loss.mean(1).sum() / num_objects


def iou_loss(inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False):
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)
    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
        use_kd_output=False,
        use_kd_intermediate=False,
        kd_loss_type="mse",
    ):
        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0
        if "loss_kd_output" not in self.weight_dict:
            self.weight_dict["loss_kd_output"] = 0.0
        if "loss_kd_intermediate" not in self.weight_dict:
            self.weight_dict["loss_kd_intermediate"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

        self.use_kd_output = use_kd_output
        self.use_kd_intermediate = use_kd_intermediate
        self.kd_loss_type = kd_loss_type

    def forward(
        self,
        outs_batch: List[Dict],
        targets_batch: torch.Tensor,
        teacher_outs_batch: List[Dict] = None,
        student_intermediates: List[List[torch.Tensor]] = None,
        teacher_intermediates: List[List[torch.Tensor]] = None,
    ):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for idx, (outs, targets) in enumerate(zip(outs_batch, targets_batch)):
            teacher_outs = teacher_outs_batch[idx] if teacher_outs_batch is not None else None
            student_inter_i = student_intermediates[idx] if student_intermediates is not None else None
            teacher_inter_i = teacher_intermediates[idx] if teacher_intermediates is not None else None
            cur_losses = self._forward(
                outs, targets, num_objects, teacher_outs, student_inter_i, teacher_inter_i
            )
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    def _forward(
        self,
        outputs: Dict,
        targets: torch.Tensor,
        num_objects,
        teacher_outputs: Dict = None,
        student_intermediates: List[torch.Tensor] = None,
        teacher_intermediates: List[torch.Tensor] = None,
    ):
        # print the range of the targets
        # print(f"targets range: {targets.min()} {targets.max()}")
        
        target_masks = targets.unsqueeze(1).float()
        assert target_masks.dim() == 4
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        if self.use_kd_output and teacher_outputs is not None:
            teacher_masks_list = teacher_outputs["multistep_pred_multimasks_high_res"]
        else:
            teacher_masks_list = [None] * len(src_masks_list)

        losses = {
            "loss_mask": 0,
            "loss_dice": 0,
            "loss_iou": 0,
            "loss_class": 0,
            "loss_kd_output": 0,
            "loss_kd_intermediate": 0,
        }

        for src_masks, ious, object_score_logits, teacher_masks in zip(
            src_masks_list, ious_list, object_score_logits_list, teacher_masks_list
        ):
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits, teacher_masks
            )

        # --- KD Loss on intermediates ---
        if self.use_kd_intermediate and student_intermediates is not None and teacher_intermediates is not None:
            for student_embed, teacher_embed in zip(student_intermediates, teacher_intermediates):
                kd_target = teacher_embed.detach()
                kd_student = student_embed

                if self.kd_loss_type == "mse":
                    loss_kd = F.mse_loss(kd_student, kd_target, reduction="mean")
                elif self.kd_loss_type == "l1":
                    loss_kd = F.l1_loss(kd_student, kd_target, reduction="mean")
                else:
                    raise ValueError(f"Unsupported kd_loss_type {self.kd_loss_type}")

                losses["loss_kd_intermediate"] += loss_kd

        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits, teacher_masks=None
    ):
        target_masks = target_masks.expand_as(src_masks)
        loss_multimask = sigmoid_focal_loss(
            src_masks, target_masks, num_objects, alpha=self.focal_alpha, gamma=self.focal_gamma, loss_on_multimask=True
        )
        loss_multidice = dice_loss(src_masks, target_masks, num_objects, loss_on_multimask=True)

        if not self.pred_obj_scores:
            loss_class = torch.tensor(0.0, dtype=loss_multimask.dtype, device=loss_multimask.device)
            target_obj = torch.ones(loss_multimask.shape[0], 1, dtype=loss_multimask.dtype, device=loss_multimask.device)
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[..., None].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits, target_obj, num_objects, alpha=self.focal_alpha_obj_score, gamma=self.focal_gamma_obj_score
            )

        loss_multiiou = iou_loss(
            src_masks, target_masks, ious, num_objects, loss_on_multimask=True, use_l1_loss=self.iou_use_l1_loss
        )

        if loss_multimask.size(1) > 1:
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"] + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        loss_mask = loss_mask * target_obj
        loss_dice = loss_dice * target_obj
        loss_iou = loss_iou * target_obj

        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

        if self.use_kd_output and teacher_masks is not None:
            kd_target = torch.sigmoid(teacher_masks.detach())
            kd_student = torch.sigmoid(src_masks)
            if self.kd_loss_type == "mse":
                loss_kd = F.mse_loss(kd_student, kd_target, reduction="mean")
            elif self.kd_loss_type == "l1":
                loss_kd = F.l1_loss(kd_student, kd_target, reduction="mean")
            else:
                raise ValueError(f"Unsupported kd_loss_type {self.kd_loss_type}")

            losses["loss_kd_output"] += loss_kd

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        return reduced_loss