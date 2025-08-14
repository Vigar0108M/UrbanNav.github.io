import os
import random
import numpy as np
import pickle as pkl
from tqdm import tqdm

from decord import VideoReader, cpu

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from scipy.spatial.transform import Rotation


def process_frames(frames, target_size):

        frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0  # Corrected normalization
    
        desired_height = target_size[0]
        desired_width = target_size[1]
        _, _, height, width = frames.shape

        pad_height = desired_height - height
        pad_width = desired_width - width
        
        # Only pad if necessary
        if pad_height > 0 or pad_width > 0:
            # Calculate padding for each side (left, right, top, bottom)
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            # Apply padding
            frames = TF.pad(
                frames, 
                (pad_left, pad_top, pad_right, pad_bottom),
            )
        elif pad_height < 0  or pad_width < 0:
            frames = TF.center_crop(frames, (desired_height, desired_width))
        
        return frames


def pose_to_matrix(pose):

    position = pose[:3]
    rotation = Rotation.from_quat(pose[3:])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = position

    return matrix


def poses_to_matrices(poses):

    positions = poses[:, :3]
    quats = poses[:, 3:]
    rotations = Rotation.from_quat(quats)
    matrices = np.tile(np.eye(4), (poses.shape[0], 1, 1))
    matrices[:, :3, :3] = rotations.as_matrix()
    matrices[:, :3, 3] = positions

    return matrices


def latlon_to_local(lat, lon, lat0, lon0):
    R_earth = 6378137  # Earth's radius in meters
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad
    x = dlon * np.cos((lat_rad + lat0_rad) / 2) * R_earth
    y = dlat * R_earth
    return x, y