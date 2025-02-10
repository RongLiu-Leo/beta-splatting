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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import (
    cameraList_from_camInfos,
    camera_to_JSON,
    transform_cameras,
    transform_points,
    similarity_from_cameras,
    align_principle_axes,
)
import torch
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from fused_ssim import fused_ssim
from tqdm import tqdm
import numpy as np


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        center_and_z_up=False,
    ):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.best_psnr = 0

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, init_type=args.init_type
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(
                scene_info.train_cameras
            )  # Multi-res consistent random shuffling
            random.shuffle(
                scene_info.test_cameras
            )  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        if center_and_z_up:

            def extract_camtoworlds(camera_list):
                w2c_mats = []
                for cam in camera_list:
                    # Construct a 4x4 world-to-camera matrix from R and T
                    w2c = np.eye(4)
                    w2c[:3, :3] = cam.R.transpose()
                    w2c[:3, 3] = cam.T
                    w2c_mats.append(w2c)
                w2c_mats = np.stack(w2c_mats, axis=0)
                # Invert to get camera-to-world matrices
                camtoworlds = np.linalg.inv(w2c_mats)
                return camtoworlds

            def update_camera_infos(camera_list, new_camtoworlds):
                for cam, new_pose in zip(camera_list, new_camtoworlds):
                    w2c = np.linalg.inv(new_pose)
                    cam.R = w2c[:3, :3].transpose()  # transpose to maintain consistency
                    cam.T = w2c[:3, 3]

            points = scene_info.point_cloud.points

            camtoworlds_train = extract_camtoworlds(scene_info.train_cameras)

            T1 = similarity_from_cameras(camtoworlds_train)

            camtoworlds_train = transform_cameras(T1, camtoworlds_train)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)

            camtoworlds_train = transform_cameras(T2, camtoworlds_train)
            points = transform_points(T2, points)

            update_camera_infos(scene_info.train_cameras, camtoworlds_train)

            if scene_info.test_cameras:
                camtoworlds_test = extract_camtoworlds(scene_info.test_cameras)
                camtoworlds_test = transform_cameras(T1, camtoworlds_test)
                camtoworlds_test = transform_cameras(T2, camtoworlds_test)
                update_camera_infos(scene_info.test_cameras, camtoworlds_test)
            scene_info.point_cloud.points = points

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    @torch.no_grad()
    def save_best_model(self):
        psnr_test = 0.0
        test_view_stack = self.getTestCameras()
        for idx, viewpoint in enumerate(test_view_stack):
            image = torch.clamp(self.gaussians.render(viewpoint)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean()
        psnr_test /= len(test_view_stack)
        if psnr_test > self.best_psnr:
            print(f"save best model. PSNR: {psnr_test}")
            self.save("best")
            self.best_psnr = psnr_test
            return True
        else:
            return False

    @torch.no_grad()
    def eval(self):
        torch.cuda.empty_cache()
        psnr_test = 0.0
        ssim_test = 0.0
        lpips_test = 0.0
        test_view_stack = self.getTestCameras()
        for idx, viewpoint in tqdm(enumerate(test_view_stack)):
            image = torch.clamp(self.gaussians.render(viewpoint)["render"], 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean()
            ssim_test += fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean()
            lpips_test += lpips(image, gt_image, net_type="vgg").mean()
        psnr_test /= len(test_view_stack)
        ssim_test /= len(test_view_stack)
        lpips_test /= len(test_view_stack)

        result = {
            "ours_best": {
                "SSIM": ssim_test.item(),
                "PSNR": psnr_test.item(),
                "LPIPS": lpips_test.item(),
            }
        }
        with open(
            os.path.join(self.model_path, "point_cloud/iteration_best/metrics.json"),
            "w",
        ) as f:
            json.dump(result, f, indent=True)
