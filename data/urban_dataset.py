import os
import random
import numpy as np

import json
import lmdb
import pickle as pkl
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from data.data_utils import pose_to_matrix, poses_to_matrices


class UrbanNavDataset(Dataset):
    def __init__(self, config: dict, mode: str, data_list: list):
        super().__init__()

        self.mode = mode
        self.config = config

        self.data_dir = config["data"]["data_dir"]
        self.video_fps = config["data"]["video_fps"]
        self.pose_fps = config["data"]["pose_fps"]
        self.target_fps = config["data"]["target_fps"]
        self.image_size = config["data"]["image_size"]
        self.arrived_prob = config["data"]["arrived_prob"]

        self.frame_multiplier = self.video_fps // self.target_fps
        self.pose_multiplier = self.pose_fps // self.target_fps

        self.context_size = config["context_size"]
        self.len_traj_pred = config["len_traj_pred"]

        self.input_noise = config["data"]["input_noise"]
        self.search_lower_bound = config["data"]["search_lower_bound"]
        self.search_upper_bound = config["data"]["search_upper_bound"]

        traj_list = data_list

        with open(config["data"]["lut_file"], 'r', encoding='utf-8') as f:
            self.scene_type_lut = json.load(f)

        if self.config["data"]["use_image"]:
            self.use_image = True
            self.image_file = self.config["data"]["image_file"]
            self.transform = transforms.Compose([
                transforms.Resize(config["data"]["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.use_image = False
            
        
        # Build look up table
        self.poses = {}
        self.labels = {}
        self.lut = []
        for traj_id in tqdm(traj_list, desc=f'Building {self.mode} dataset', ncols=120):
            pose_data = {}
            pose_path = os.path.join(self.data_dir, traj_id, traj_id + '.txt')
            label_path = os.path.join(self.data_dir, traj_id, traj_id + '.json')
            
            # Pose filter
            pose = np.loadtxt(pose_path, delimiter=" ")[::max(1, self.pose_multiplier), 1:]
            pose_nan = np.isnan(pose).any(axis=1)
            if np.any(pose_nan):
                first_nan_idx = np.argmin(pose_nan)
                pose = pose[:first_nan_idx]
            usable = pose.shape[0] - self.context_size - max(self.search_lower_bound, self.len_traj_pred)
            if usable < 0:
                continue

            pose_data["count"] = usable
            pose_data["step_scale"] = np.linalg.norm(np.diff(pose[:, [0, 2]], axis=0), axis=1).mean()
            pose_data["pose"] = pose
            self.poses[traj_id] = pose_data
            
            with open(label_path, 'r', encoding='utf-8') as file:
                label_data = json.load(file)
            self.labels[traj_id] = label_data
            
            for label_idx in label_data.keys():
                label_idx = int(label_idx)
                if label_idx > self.context_size + self.search_lower_bound and label_idx < pose.shape[0]:
                    self.lut.append((traj_id, label_idx))
        
        self._load_visual_feat()
        print(f"Waypoints number: {len(self.lut)}")


    def transform_poses(self, poses, current_pose_array):

        current_pose_matrix = pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)

        pose_matrices = poses_to_matrices(poses)
        transformed_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], pose_matrices)
        positions = transformed_matrices[:, :3, 3]

        return positions


    def transform_target_pose(self, target_pose, current_pose_array):

        current_pose_matrix = pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)

        target_pose_matrix = pose_to_matrix(target_pose)
        transformed_target_matrix = np.matmul(current_pose_inv, target_pose_matrix)
        target_position = transformed_target_matrix[:3, 3]

        return target_position
    

    def _load_visual_feat(self):
        # self.visual_feat = lmdb.Environment = lmdb.open(self.config["data"]["feat_file"], readonly=True, lock=False)
        self.visual_feat = lmdb.open(self.config["data"]["feat_file"], readonly=True, lock=False)


    def __getstate__(self):
        state = self.__dict__.copy()
        state["visual_feat"] = None
        return state
    

    def __setstate__(self, state):
        self.__dict__ = state
        if self.config["data"]["feat_file"] is not None:
            self._load_visual_feat()
    

    def __len__(self):
        return len(self.lut)
    

    def __getitem__(self, index):

        traj_id, target_idx = self.lut[index]

        # Get label data
        label = self.labels[traj_id]
        instr_data = random.choice(label[str(target_idx).zfill(4)])
        instr = instr_data["prompt"]
        bbox = instr_data["bbox"]
        distance = instr_data["distance"]
        if distance < 5:
            arrived = (random.random() < self.arrived_prob)
        else:
            arrived = False

        # Sample start indices
        if arrived:
            pose_start_idx = target_idx - self.context_size - self.len_traj_pred
        else:
            
            lower_bound = max(target_idx - self.search_upper_bound, self.context_size)
            upper_bound = target_idx - self.search_lower_bound
            pose_start_idx = random.randint(lower_bound, upper_bound) - self.context_size

        # Get frame indices
        frame_start_idx = pose_start_idx * self.frame_multiplier
        frame_indices = frame_start_idx + np.arange(self.context_size + self.len_traj_pred) * self.frame_multiplier

        # Load frames and features
        features = []
        with self.visual_feat.begin() as txn:
            for frame_idx in frame_indices:
                while True:
                    key = traj_id + '_' + str(frame_idx).zfill(4)
                    value = txn.get(key.encode('ascii'))
                    if value:
                        feature = pkl.loads(value)
                        features.append(feature)
                        break
                    else:
                        frame_idx -= self.frame_multiplier

        all_feat = np.stack(features, axis=0)
        obs_feat = torch.from_numpy(all_feat[:self.context_size])
        future_feat = torch.from_numpy(all_feat[self.context_size:])

        if self.use_image:
            curr_idx = frame_indices[self.context_size - 1]
            scene = traj_id.rsplit("_", 1)[0]
            scene_type = self.scene_type_lut[scene]
            curr_frame_path = os.path.join(self.image_file, scene_type, scene, traj_id, str(curr_idx).zfill(4) + '.jpg')
            curr_frame = self.transform(Image.open(curr_frame_path).convert("RGB"))

        # Get pose data
        current_pose_idx = pose_start_idx + self.context_size
        pose = self.poses[traj_id]["pose"]
        input_poses = pose[pose_start_idx: current_pose_idx]
        assert int(target_idx) > current_pose_idx
        
        # Get target pose and waypoint poses
        target_pose = pose[int(target_idx)]
        waypoint_poses = pose[current_pose_idx: current_pose_idx + self.len_traj_pred]

        # Transform poses
        current_pose = input_poses[-1]
        transformed_input_poses = self.transform_poses(input_poses, current_pose)[:, [0, 2]]
        target_pose = self.transform_target_pose(target_pose, current_pose)[[0, 2]]
        transformed_waypoint_poses = self.transform_poses(waypoint_poses, current_pose)

        # Convert data to tensors
        step_scale = torch.tensor(self.poses[traj_id]["step_scale"], dtype=torch.float32)
        step_scale = torch.clamp(step_scale, min=1e-2)

        input_poses = torch.tensor(transformed_input_poses, dtype=torch.float32) / step_scale + torch.randn(self.context_size, 2) * self.input_noise
        target_pose = torch.tensor(target_pose, dtype=torch.float32)
        waypoint_poses = torch.tensor(transformed_waypoint_poses[:, [0, 2]], dtype=torch.float32) / step_scale

        arrived = torch.tensor(arrived, dtype=torch.float32)

        sample = {
            'instr': instr,
            'bbox': bbox,
            'obs_feat': obs_feat,
            'future_feat': future_feat,
            'input_poses': input_poses,
            'target_pose': target_pose,
            'waypoint_poses': waypoint_poses,
            'step_scale': step_scale,
            'arrived': arrived
        }
        if self.use_image:
            sample['curr_frame'] = curr_frame

        return sample

