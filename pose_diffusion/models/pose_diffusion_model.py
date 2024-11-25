# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Standard library imports
import base64
import io
import logging
import math
import pickle
import warnings
from collections import defaultdict
from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party library imports
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import se3_exp_map, se3_log_map, Transform3d, so3_relative_angle
from util.camera_transform import pose_encoding_to_camera, camera_to_pose_encoding

import models
from hydra.utils import instantiate
from pytorch3d.renderer.cameras import PerspectiveCameras

from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

logger = logging.getLogger(__name__)


class PoseDiffusionModel(nn.Module):
    def __init__(self, pose_encoding_type: str, IMAGE_FEATURE_EXTRACTOR: Dict, DIFFUSER: Dict, DENOISER: Dict):
        """Initializes a PoseDiffusion model.

        Args:
            pose_encoding_type (str):
                Defines the encoding type for extrinsics and intrinsics
                Currently, only `"absT_quaR_logFL"` is supported -
                a concatenation of the translation vector,
                rotation quaternion, and logarithm of focal length.
            image_feature_extractor_cfg (Dict):
                Configuration for the image feature extractor.
            diffuser_cfg (Dict):
                Configuration for the diffuser.
            denoiser_cfg (Dict):
                Configuration for the denoiser.
        """

        super().__init__()

        self.pose_encoding_type = pose_encoding_type

        self.image_feature_extractor = instantiate(IMAGE_FEATURE_EXTRACTOR, _recursive_=False)
        self.diffuser = instantiate(DIFFUSER, _recursive_=False)

        denoiser = instantiate(DENOISER, _recursive_=False)
        self.diffuser.model = denoiser

        self.target_dim = denoiser.target_dim

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        image: torch.Tensor,
        gt_cameras: Optional[CamerasBase] = None,
        sequence_name: Optional[List[str]] = None,
        cond_fn=None,
        cond_start_step=0,
        training=True,
        batch_repeat=-1,
    ):
        """
        Forward pass of the PoseDiffusionModel.

        Args:
            image (torch.Tensor):
                Input image tensor, Bx3xHxW.
            gt_cameras (Optional[CamerasBase], optional):
                Camera object. Defaults to None.
            sequence_name (Optional[List[str]], optional):
                List of sequence names. Defaults to None.
            cond_fn ([type], optional):
                Conditional function. Wrapper for GGS or other functions.
            cond_start_step (int, optional):
                The sampling step to start using conditional function.

        Returns:
            PerspectiveCameras: PyTorch3D camera object.
        """

        shapelist = list(image.shape)
        batch_num = shapelist[0]
        frame_num = shapelist[1]

        reshaped_image = image.reshape(batch_num * frame_num, *shapelist[2:])
        z = self.image_feature_extractor(reshaped_image).reshape(batch_num, frame_num, -1)
        if training:
            pose_encoding = camera_to_pose_encoding(gt_cameras, pose_encoding_type=self.pose_encoding_type)

            if batch_repeat > 0:
                pose_encoding = pose_encoding.reshape(batch_num * batch_repeat, -1, self.target_dim)
                z = z.repeat(batch_repeat, 1, 1)
            else:
                pose_encoding = pose_encoding.reshape(batch_num, -1, self.target_dim)

            diffusion_results = self.diffuser(pose_encoding, z=z)

            # Begin Ramana edit
            pair_idx_i1, pair_idx_i2 = batched_all_pairs(batch_num, frame_num, z.device)
            pred_pose_enc = diffusion_results['x_0_pred'].reshape(batch_num*frame_num, -1)[:, :7]
            gt_se3_flat = gt_cameras.get_world_to_view_transform().get_matrix()
            pred_R, pred_t, pred_se3_flat = pose_encoding_to_R_t(pred_pose_enc)
            rel_pose_from_pred = closed_form_inverse(pred_se3_flat[pair_idx_i1]).bmm(pred_se3_flat[pair_idx_i2])
            rel_pose_from_gt = closed_form_inverse(gt_se3_flat[pair_idx_i1]).bmm(gt_se3_flat[pair_idx_i2])

            R12 = torch.bmm(rel_pose_from_pred[:, :3, :3], rel_pose_from_gt[:, :3, :3].permute(0, 2, 1))
            r_loss =  (R12 - torch.eye(3).to(R12.device).unsqueeze(0).clone()) ** 2
            t_loss = (rel_pose_from_pred[:, 3, :3] - rel_pose_from_gt[:, 3, :3]) ** 2
            loss = r_loss.mean() + t_loss.mean()
            diffusion_results["loss"] = 0.1*loss
            # End Ramana Edit
            # import pdb;pdb.set_trace()

            diffusion_results["pred_cameras"] = pose_encoding_to_camera(
                diffusion_results["x_0_pred"], pose_encoding_type=self.pose_encoding_type
            )

            return diffusion_results
        else:
            B, N, _ = z.shape

            target_shape = [B, N, self.target_dim]

            # sampling
            (pose_encoding, pose_encoding_diffusion_samples) = self.diffuser.sample(
                shape=target_shape, z=z, cond_fn=cond_fn, cond_start_step=cond_start_step
            )

            # convert the encoded representation to PyTorch3D cameras
            pred_cameras = pose_encoding_to_camera(pose_encoding, pose_encoding_type=self.pose_encoding_type)

            diffusion_results = {"pred_cameras": pred_cameras, "z": z}

            return diffusion_results


def closed_form_inverse(se3):
    # se3:    Nx4x4
    # return: Nx4x4
    # inverse each 4x4 matrix
    R = se3[:,:3,:3]
    T = se3[:, 3:, :3]
    R_trans = R.transpose(1,2)

    left_down = - T.bmm(R_trans)
    left = torch.cat((R_trans,left_down),dim=1)
    right = se3[:,:,3:].detach().clone()
    inversed = torch.cat((left,right),dim=-1)
    return inversed

def pose_encoding_to_R_t(pose_encoding):
    transform = torch.zeros(pose_encoding.shape[0], 4, 4, dtype=pose_encoding.dtype, device=pose_encoding.device)
    abs_T = pose_encoding[:, :3]
    quaternion_R = quaternion_to_matrix(pose_encoding[:, 3:7])
    transform[:, :3, :3] = quaternion_R
    transform[:, 3, :3] = abs_T
    transform[:, 3, 3] = 1.0
    return quaternion_R, abs_T, transform

def batched_all_pairs(B, N, device):
    # se3 in B x N x 4 x 4
    i1_, i2_ = torch.combinations(
        torch.arange(N), 2, with_replacement=False
    ).to(device).unbind(-1)
    i1, i2 = [
        (i[None] + torch.arange(B, device=device)[:, None] * N).reshape(-1)
        for i in [i1_, i2_]
    ]
    return i1, i2
