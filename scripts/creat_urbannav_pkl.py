import os
import json
import random
import pickle
from pathlib import Path

random.seed(0)

def random_sample(lst: list, ratio=0.1):
    if not lst:
        return []
    count = max(1, int(len(lst) * ratio))
    return random.sample(lst, count)

def random_split_list(lst, train_ratio=0.8):
    random.shuffle(lst)
    split_index = int(len(lst) * train_ratio)
    return lst[:split_index], lst[split_index:]


if __name__ == "__main__":
    
    root_path = "/path/to/datas"
    output_path = "/path/to/datasets"
    scene_type_lut_path = "/path/to/scene_type_lut.json"
    test_ratio = 0.2

    with open(scene_type_lut_path, 'r', encoding='utf-8') as f:
        scene_type_lut = json.load(f)
    all_scene_list = list(scene_type_lut.keys())
    unseen_scene_list = random_sample(all_scene_list, test_ratio / 2)

    train_traj_list = []
    test_seen_traj_list = []
    test_unseen_traj_list = []
    traj_list = [traj_folder.name for traj_folder in Path(root_path).iterdir() if traj_folder.is_dir()]

    for traj_id in traj_list:
        scene = traj_id.rsplit('_', 1)[0]
        if scene in unseen_scene_list:
            test_unseen_traj_list.append(traj_id)
        else:
            random_flag = random.random()
            if random_flag <= 1 - test_ratio / 2:
                train_traj_list.append(traj_id)
            else:
                test_seen_traj_list.append(traj_id)

    # Save
    with open(os.path.join(output_path, 'landmark_train_data.pkl'), 'wb') as f:
        pickle.dump(train_traj_list, f)
    with open(os.path.join(output_path, 'landmark_test_seen_data.pkl'), 'wb') as f:
        pickle.dump(test_seen_traj_list, f)
    with open(os.path.join(output_path, 'landmark_test_unseen_data.pkl'), 'wb') as f:
        pickle.dump(test_unseen_traj_list, f)

    print(f"train data: {len(train_traj_list)}")
    print(f"test seen data: {len(test_seen_traj_list)}")
    print(f"test unseen data: {len(test_unseen_traj_list)}")
    