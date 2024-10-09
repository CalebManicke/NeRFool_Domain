# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import imageio
import torch
import sys
sys.path.append('../')
from torch.utils.data import Dataset
from .data_utils import random_crop, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses

def createCamCoordinates(n :int, m : int, px : int , py: int) -> np.ndarray:
    '''
    Generate the camera coordinate on the x, y plane
    Input:
        m n: row column (number of elemental image in vertical and horizontal direction) type: int
        px, py :pitch (in mm) type: int
    Output: 
        np.ndarray of shape (2, m*n) , row by row, from left top
    '''

    cameraCoordinates = np.zeros((2,m*n))
    cameraCoordinates[0, :] = np.tile(np.arange(0, n) , m) * px
    cameraCoordinates[1, :] = np.arange(0, m).repeat(n) * py

    return cameraCoordinates.T

def singlePoseGen(TxTy : np.ndarray, hwf: tuple) -> np.ndarray:
    '''
    convert custom format pose to 4x4 matrix of intrinsics and extrinsics (opencv convention)
    Args:
        pose: matrix [3, 4]
    Returns: intrinsics [4, 4] and c2w [4, 4]
    '''
    (h, w, f) = hwf

    sx  =  36 # sensor size in mm in x direction
    sy  =  24 # sensor size in mm in y direction

    c2w_4x4 = np.eye(4)
    c2w_4x4[2,3] = 0 #set Tz to 0
    c2w_4x4[0:2,3] = TxTy
    # c2w_4x4[:, 1:3] *= -1
    intrinsics = np.array([[w*f/sx,   0     ,   w / 2., 0],
                           [0     ,  h*f/sy ,   h / 2., 0],
                           [0     ,   0     ,     1   , 0],
                           [0     ,   0     ,     0   , 1]])
    return intrinsics, c2w_4x4

def generatePoses(cameraCoordinates : np.ndarray, hwf: tuple) -> np.ndarray:
    all_intrinsics = []
    all_c2w_mats = []
    for TxTy in cameraCoordinates:
        # print(TxTy)
        intrinsics, c2w_mat = singlePoseGen(TxTy, hwf)
        all_intrinsics.append(intrinsics)
        all_c2w_mats.append(c2w_mat)
    all_intrinsics = np.stack(all_intrinsics)
    all_c2w_mats = np.stack(all_c2w_mats)
    return all_intrinsics, all_c2w_mats

def _load_image_files(imgdir):
    imgfiles = {int(f[4:-4]) : os.path.join(imgdir, f) for f in os.listdir(imgdir) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')}
    # sorted image file
    total_imgs = len(imgfiles)
    temp_files = []
    for i in range(total_imgs):
        temp_files.append(imgfiles[i + 1])
    imgfiles = temp_files

    return imgfiles



def load_integral_imaging_data(scene_path, load_imgs=False, factor=None):
    imgfiles = _load_image_files(scene_path)


    def imread(f):
            if f.endswith('png'):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)

    if not load_imgs:
        imgs = None
    else:
        imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
        imgs = np.stack(imgs, -1)
    img0 = imread(imgfiles[0])
    cameraCoordinates = createCamCoordinates(n = 6,m = 6,px = 10,py = 10)
    poses = generatePoses(cameraCoordinates, (*img0.shape[:2], 50))

        # print('Loaded image data', imgs.shape, poses[:, -1, 0])
    bounds = (250,400) # just a guess work for the time being
    return imgs, poses,  bounds, None, None, imgfiles

class IntegralImagingDataset(Dataset):
    def __init__(self, args, mode, scenes=(), use_glb_src=False, **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'data/integral_imaging/')
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = args.random_crop
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        
        self.train_depth_files = []
        self.render_depth_files = []
        
        self.test_poses = []
        
        self.use_glb_src = use_glb_src

        all_scenes = os.listdir(self.folder_path)
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes
        
        # assert mode == "train",  "Only support training mode for Integral Imaging dataset"

        print("loading {} for {}".format(scenes, mode))
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(self.folder_path, scene)
            _, poses, bds, _, i_test, rgb_files = load_integral_imaging_data(scene_path, load_imgs=False, factor=args.llff_factor)
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = poses
            i_test = np.arange(0,0)[::self.args.llffhold] # throw everthing in train dataset
            i_train = np.array([j for j in np.arange(int(intrinsics.shape[0])) if
                                (j not in i_test)])

            # if mode == 'train':
            #     i_render = i_train
            # else:
            #     i_render = i_test
            i_render = i_train
                # raise NotImplementedError("Test mode not implemented for Integral Imaging dataset")
            
            self.test_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_test]])

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        idx = idx % len(self.render_rgb_files)
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        depth_range = torch.tensor([depth_range[0], depth_range[1]])
        
        data = {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'depth_range': depth_range,
                # 'depth': None if len(self.render_depth_files)==0 else torch.from_numpy(depth),
                # 'src_depths': None if len(self.train_depth_files)==0 else torch.from_numpy(src_depths)
                }
        
        return data
