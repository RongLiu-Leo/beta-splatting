#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from gsplat.rendering import rasterization 
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from matplotlib import pyplot as plt

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    beta = pc.get_beta

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling * scaling_modifier
        rotations = pc.get_rotation
        
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_shs
    sb_params = pc.get_sb_params
    
    # Convert OpenGL 4x4 projection matrix to 3x3 intrinsic matrix format
    K = torch.zeros((3,3), device=viewpoint_camera.projection_matrix.device)
    
    fx = 0.5 * viewpoint_camera.image_width / math.tan(viewpoint_camera.FoVx / 2)
    fy = 0.5 * viewpoint_camera.image_height / math.tan(viewpoint_camera.FoVy / 2)
    
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = viewpoint_camera.image_width / 2
    K[1,2] = viewpoint_camera.image_height / 2
    K[2,2] = 1.0

    # call the gsplat renderer
    rgbs, alphas, meta = rasterization(
        means=means3D,
        quats=rotations,
        scales=scales,
        opacities=opacity.squeeze(),
        betas=beta.squeeze(),
        colors=shs,
        viewmats=viewpoint_camera.world_view_transform.transpose(0,1).unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=viewpoint_camera.image_width,
        height=viewpoint_camera.image_height,
        backgrounds=bg_color.unsqueeze(0),
        render_mode="RGB",
        covars=cov3D_precomp,
        sh_degree=pc.active_sh_degree,
        sb_number=pc.sb_number,
        sb_params=sb_params,
        packed=False,
    )
    
    # # Convert from N,H,W,C to N,C,H,W format
    rgbs = rgbs.permute(0, 3, 1, 2).contiguous()[0]
    
    return {"render": rgbs,
            "viewspace_points": meta["means2d"],
            "visibility_filter" : meta["radii"] > 0,
            "radii": meta["radii"],
            "is_used": meta["radii"] > 0}