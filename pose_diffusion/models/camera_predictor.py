# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from dataclasses import field, dataclass

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from einops import rearrange, repeat
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.renderer import PerspectiveCameras

# from genai3d.models.pose_pred_modules import *
from pose_pred_modules import *

from pytorch3d.renderer.implicit.harmonic_embedding import HarmonicEmbedding
from pytorch3d.transforms.so3 import so3_relative_angle


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


logger = logging.getLogger(__name__)

class PoseEmbedding(nn.Module):
    def __init__(self, target_dim, n_harmonic_functions=10, append_input=True):
        super().__init__()

        self._emb_pose = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_functions, append_input=append_input
        )

        self.out_dim = self._emb_pose.get_output_dim(target_dim)

    def forward(self, pose_encoding):
        e_pose_encoding = self._emb_pose(pose_encoding)
        return e_pose_encoding
    

def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: Union[int, Tuple[int, int]], return_grid=False
) -> torch.Tensor:
    """
    This function initializes a grid and generates a 2D positional embedding using sine and cosine functions.
    It is a wrapper of get_2d_sincos_pos_embed_from_grid.
    Args:
    - embed_dim: The embedding dimension.
    - grid_size: The grid size.
    Returns:
    - pos_embed: The generated 2D positional embedding.
    """
    if isinstance(grid_size, tuple):
        grid_size_h, grid_size_w = grid_size
    else:
        grid_size_h = grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float)
    grid_w = torch.arange(grid_size_w, dtype=torch.float)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if return_grid:
        return (
            pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(
                0, 3, 1, 2
            ),
            grid,
        )
    return pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(
        0, 3, 1, 2
    )


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb[None].float()
    

def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - grid: The grid to generate the embedding from.

    Returns:
    - emb: The generated 2D positional embedding.
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=2)  # (H*W, D)
    return emb

def pose_encoding_to_R_t(pose_encoding):
    transform = torch.zeros(pose_encoding.shape[0], 4, 4, dtype=pose_encoding.dtype, device=pose_encoding.device)
    abs_T = pose_encoding[:, :3]
    quaternion_R = quaternion_to_matrix(pose_encoding[:, 3:7])
    transform[:, :3, :3] = quaternion_R
    transform[:, 3, :3] = abs_T
    transform[:, 3, 3] = 1.0
    return quaternion_R, abs_T, transform



def camera_to_pose_encoding(
    camera,
    pose_encoding_type="absT_quaR_OneFL",
    log_focal_length_bias=1.8,
    min_focal_length=0.1,
    max_focal_length=30,
):
    """
    Inverse to pose_encoding_to_camera
    """
    if pose_encoding_type == "absT_quaR_logFL":
        # Convert rotation matrix to quaternion
        quaternion_R = matrix_to_quaternion(camera.R)

        # Calculate log_focal_length
        log_focal_length = (
            torch.log(
                torch.clamp(
                    camera.focal_length,
                    min=min_focal_length,
                    max=max_focal_length,
                )
            )
            - log_focal_length_bias
        )

        # Concatenate to form pose_encoding
        pose_encoding = torch.cat(
            [camera.T, quaternion_R, log_focal_length], dim=-1
        )

    elif pose_encoding_type == "absT_quaR_OneFL":
        # [absolute translation, quaternion rotation, normalized focal length]
        quaternion_R = matrix_to_quaternion(camera.R)
        focal_length = (
            torch.clamp(
                camera.focal_length, min=min_focal_length, max=max_focal_length
            )
        )[..., 0:1]
        pose_encoding = torch.cat(
            [camera.T, quaternion_R, focal_length], dim=-1
        )
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    return pose_encoding



def pose_encoding_to_camera(
    pose_encoding,
    pose_encoding_type="absT_quaR_logFL",
    log_focal_length_bias=1.8,
    min_focal_length=0.1,
    max_focal_length=30,
    return_dict=False,
    to_OpenCV=False,
):
    """
    Args:
        pose_encoding: A tensor of shape `BxNxC`, containing a batch of
                        `BxN` `C`-dimensional pose encodings.
        pose_encoding_type: The type of pose encoding,
    """
    pose_encoding_reshaped = pose_encoding.reshape(
        -1, pose_encoding.shape[-1]
    )  # Reshape to BNxC

    if pose_encoding_type == "absT_quaR_logFL":
        # 3 for absT, 4 for quaR, 2 for absFL
        abs_T = pose_encoding_reshaped[:, :3]
        quaternion_R = pose_encoding_reshaped[:, 3:7]
        R = quaternion_to_matrix(quaternion_R)
        log_focal_length = pose_encoding_reshaped[:, 7:9]
        # log_focal_length_bias was the hyperparameter
        # to ensure the mean of logFL close to 0 during training
        # Now converted back
        focal_length = (log_focal_length + log_focal_length_bias).exp()
        # clamp to avoid weird fl values
        focal_length = torch.clamp(
            focal_length, min=min_focal_length, max=max_focal_length
        )
    elif pose_encoding_type == "absT_quaR_OneFL":
        # 3 for absT, 4 for quaR, 1 for absFL
        # [absolute translation, quaternion rotation, normalized focal length]
        abs_T = pose_encoding_reshaped[:, :3]
        quaternion_R = pose_encoding_reshaped[:, 3:7]
        R = quaternion_to_matrix(quaternion_R)
        focal_length = pose_encoding_reshaped[:, 7:8]
        focal_length = torch.clamp(
            focal_length, min=min_focal_length, max=max_focal_length
        )
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    if to_OpenCV:
        ### From Pytorch3D coordinate to OpenCV coordinate:
        # I hate coordinate conversion
        R = R.clone()
        abs_T = abs_T.clone()
        R[:, :, :2] *= -1
        abs_T[:, :2] *= -1
        R = R.permute(0, 2, 1)

        extrinsics_4x4 = torch.eye(4, 4).to(R.dtype).to(R.device)
        extrinsics_4x4 = extrinsics_4x4[None].repeat(len(R), 1, 1)

        extrinsics_4x4[:, :3, :3] = R.clone()
        extrinsics_4x4[:, :3, 3] = abs_T.clone()

        rel_transform = closed_form_inverse_OpenCV(extrinsics_4x4[0:1])
        rel_transform = rel_transform.expand(len(extrinsics_4x4), -1, -1)

        # relative to the first camera
        # NOTE it is extrinsics_4x4 x rel_transform instead of rel_transform x extrinsics_4x4
        # this is different in opencv / pytorch3d convention
        extrinsics_4x4 = torch.bmm(extrinsics_4x4, rel_transform)

        R = extrinsics_4x4[:, :3, :3].clone()
        abs_T = extrinsics_4x4[:, :3, 3].clone()

    if return_dict:
        return {"focal_length": focal_length, "R": R, "T": abs_T}

    pred_cameras = PerspectiveCameras(
        focal_length=focal_length, R=R, T=abs_T, device=R.device
    )
    return pred_cameras

class CameraPredictor(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_heads=8,
        mlp_ratio=4,
        z_dim: int = 768,
        down_size=336,
        att_depth=8,
        trunk_depth=4,
        backbone="dinov2b",
        pose_encoding_type="absT_quaR_OneFL",
        out_dim=8,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg

        self.att_depth = att_depth
        self.down_size = down_size
        self.pose_encoding_type = pose_encoding_type

        # if self.pose_encoding_type == "absT_quaR_OneFL":
        #     self.target_dim = 8
        # if self.pose_encoding_type == "absT_quaR_logFL":
        #     self.target_dim = 9
        self.target_dim = out_dim

        self.backbone = self.get_backbone(backbone)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.input_transform = Mlp(
            in_features=z_dim, out_features=hidden_size, drop=0
        )
        self.norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        # sine and cosine embed for camera parameters
        self.embed_pose = PoseEmbedding(
            target_dim=self.target_dim,
            n_harmonic_functions=(hidden_size // self.target_dim) // 2,
            append_input=False,
        )

        self.pose_token = nn.Parameter(
            torch.zeros(1, 1, 1, hidden_size)
        )  # register

        self.pose_branch = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size * 2,
            out_features=hidden_size + self.target_dim,
            drop=0,
        )

        self.ffeat_updater = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU()
        )

        self.self_att = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=nn.MultiheadAttention,
                )
                for _ in range(self.att_depth)
            ]
        )

        self.trunk = nn.Sequential(
            *[
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=nn.MultiheadAttention,
                )
                for _ in range(trunk_depth)
            ]
        )

        self.gamma = 0.8

        nn.init.normal_(self.pose_token, std=1e-6)

        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 3, 1, 1),
                persistent=False,
            )

    def forward(
        self,
        image,
        gt_cameras,
        iters=4,        
        training=True, # dummy
        batch_repeat=0, # dummy
    ):
        """
        reshaped_image: BxFx3xHxW. The values of reshaped_image are within [0, 1]
        preliminary_cameras: cameras in opencv coordinate.
        """
        
        shapelist = list(image.shape)
        batch_num = shapelist[0]
        frame_num = shapelist[1]
        reshaped_image = image.reshape(batch_num * frame_num, *shapelist[2:])

        rgb_feat, B, S, C = self.get_2D_image_features(
            reshaped_image, batch_num
        )

        # Or you can use random init for the poses
        pred_pose_enc = torch.zeros(B, S, self.target_dim).to(rgb_feat.device)

        rgb_feat_init = rgb_feat.clone()

        pose_predictions = []



        for iter_num in range(iters):
            pred_pose_enc = pred_pose_enc.detach()

            # Embed the camera parameters and add to rgb_feat
            pose_embed = self.embed_pose(pred_pose_enc)
            rgb_feat = rgb_feat + pose_embed

            # Run trunk transformers on rgb_feat
            rgb_feat = self.trunk(rgb_feat)

            # Predict the delta feat and pose encoding at each iteration
            delta = self.pose_branch(rgb_feat)
            delta_pred_pose_enc = delta[..., : self.target_dim]
            delta_feat = delta[..., self.target_dim :]

            rgb_feat = self.ffeat_updater(self.norm(delta_feat)) + rgb_feat

            pred_pose_enc = pred_pose_enc + delta_pred_pose_enc

            # Residual connection
            rgb_feat = (rgb_feat + rgb_feat_init) / 2
        
            pose_predictions.append(pred_pose_enc)

        
        gt_pose_enc = camera_to_pose_encoding(gt_cameras, self.pose_encoding_type)
        gt_pose_enc = gt_pose_enc.reshape(batch_num, frame_num, -1)

        seq_loss = sequence_loss_Pose(pose_predictions, gt_pose_enc, gamma=0.6)
        pred_cameras = pose_encoding_to_camera(pred_pose_enc.detach(), self.pose_encoding_type)

        pose_predictions = {
            "pred_pose_enc": pred_pose_enc,
            "pred_cameras": pred_cameras,
            "loss": seq_loss,
            "rgb_feat_init": rgb_feat_init,
        }

        return pose_predictions


    def get_backbone(self, backbone):
        """
        Load the backbone model.
        """
        if backbone == "dinov2s":
            return torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14_reg"
            )
        elif backbone == "dinov2b":
            return torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14_reg"
            )
        else:
            raise NotImplementedError(f"Backbone '{backbone}' not implemented")

    def _resnet_normalize_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self._resnet_mean) / self._resnet_std

    def get_2D_image_features(self, reshaped_image, batch_size):
        # Get the 2D image features
        if reshaped_image.shape[-1] != self.down_size:
            reshaped_image = F.interpolate(
                reshaped_image,
                (self.down_size, self.down_size),
                mode="bilinear",
                align_corners=True,
            )

        with torch.no_grad():
            reshaped_image = self._resnet_normalize_image(reshaped_image)
            rgb_feat = self.backbone(reshaped_image, is_training=True)
            # B x P x C
            rgb_feat = rgb_feat["x_norm_patchtokens"]

        rgb_feat = self.input_transform(rgb_feat)
        rgb_feat = self.norm(rgb_feat)

        rgb_feat = rearrange(rgb_feat, "(b s) p c -> b s p c", b=batch_size)

        B, S, P, C = rgb_feat.shape
        patch_num = int(math.sqrt(P))

        # add embedding of 2D spaces
        pos_embed = get_2d_sincos_pos_embed(
            C, grid_size=(patch_num, patch_num)
        ).permute(0, 2, 3, 1)[None]
        pos_embed = pos_embed.reshape(1, 1, patch_num * patch_num, C).to(
            rgb_feat.device
        )

        rgb_feat = rgb_feat + pos_embed

        # register for pose
        pose_token = self.pose_token.expand(B, S, -1, -1)
        rgb_feat = torch.cat([pose_token, rgb_feat], dim=-2)

        B, S, P, C = rgb_feat.shape # P here is P+1
        # Make it (B*F, P, C)
        rgb_feat = rearrange(rgb_feat, "b s p c -> (b s) p c", b=B, s=S)

        for idx in range(self.att_depth):
            # self attention
            rgb_feat = self.self_att[idx](rgb_feat)

        rgb_feat = rearrange(rgb_feat, "(b s) p c -> b s p c", b=B, s=S)

        rgb_feat = rgb_feat[:, :, 0] # B,S,C

        return rgb_feat, B, S, C
    

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



def sequence_loss_Pose(pose_preds, pose_gt, gt_track=None, gamma=0.8, cfg = None):
    """Loss function defined over sequence of pose predictions"""
    B, S, D = pose_gt.shape
    n_predictions = len(pose_preds)
    pose_loss = 0.0

    # if cfg.se3loss:
    #     pair1, pair2 = batched_all_pairs(B,S)
    #     gt_cameras = pose_encoding_to_camera(pose_gt.clone(), cfg.MODEL.pose_encoding_type)
    #     gt_se3 = gt_cameras.get_world_to_view_transform().get_matrix()    
    #     rel_gt = closed_form_inverse(gt_se3[pair1]).bmm(gt_se3[pair2])


    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        pose_pred = pose_preds[i]  # [:,:,0:1]
        i_loss = (pose_pred - pose_gt).abs()  # B, S, N, 2
        i_loss = torch.mean(i_loss)  # B, S, N

        # if cfg.se3loss:
        #     pred_cameras = pose_encoding_to_camera(pose_pred.clone(), cfg.MODEL.pose_encoding_type)
        #     pred_se3 = pred_cameras.get_world_to_view_transform().get_matrix()    
        #     rel_pred = closed_form_inverse(pred_se3[pair1]).bmm(pred_se3[pair2])
        #     relR = (matrix_to_quaternion(rel_pred[:,:3,:3]) - matrix_to_quaternion(rel_gt[:,:3,:3])) ** 2
        #     relT = (rel_pred[:,3:,:3] - rel_gt[:,3:,:3]).norm(dim=-1)
        #     relLoss = relR.mean() + relT.mean() * 0.5
        #     pose_loss += i_weight * relLoss
        # else:
        
        pose_loss += i_weight * i_loss

        # if gt_track is not None:
        #     epiloss = compute_epipolar_distance(pose_pred, cfg, pair1, pair2, leftKP, rightKP)
        #     epiloss = epiloss.mean()
        #     pose_loss += i_weight * epiloss * cfg.epiloss_w


    pose_loss = pose_loss / n_predictions
    return pose_loss

