import os
from pathlib import Path
import shutil
from tqdm import tqdm


def get_traj_list(root_folder):
    path = Path(root_folder)
    file_names = [f.stem for f in path.glob("*.json")]
    return file_names


def get_label_path(traj_id, label_folder):
    label_path = os.path.join(label_folder, traj_id + '.json')
    return label_path


def get_pose_path(traj_id, pose_folder):
    pose_path = os.path.join(pose_folder, traj_id + '.txt')
    return pose_path


if __name__ == "__main__":
    
    label_folder = "/mnt/share/yhmei/dataset/dataset_CityWalker/labels"
    pose_folder = "/mnt/share/yhmei/dataset/dataset_CityWalker/poses"
    output_folder = "/mnt/share/yhmei/dataset/dataset_LandMark/datas"

    traj_list = get_traj_list(label_folder)

    for traj_id in tqdm(traj_list, ncols=150):

        traj_folder = os.path.join(output_folder, traj_id)

        if os.path.isdir(traj_folder):
            print(f'Skip traj: {traj_id}')
        else:
            os.makedirs(traj_folder, exist_ok=False)

            label_file_path = get_label_path(traj_id, label_folder)
            pose_file_path = get_pose_path(traj_id, pose_folder)

            shutil.copy(label_file_path, traj_folder)
            shutil.copy(pose_file_path, traj_folder)

