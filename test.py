import os
import yaml
import json
import argparse
import numpy as np
from datetime import datetime
from statistics import mean

import torch
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from utils.train_utils import test_epoch
from utils.setup_utils import setup_dataset, setup_model


def test(
        config: dict,
        seen_dataset: Dataset,
        unseen_dataset: Dataset,
        device: torch.device
):

    # Set random seed
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    # Build dataloader
    batch_size = config["batch_size"]
    seen_loader = DataLoader(
        seen_dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        shuffle=False,
        drop_last=True,
        persistent_workers=True
    )
    unseen_loader = DataLoader(
        unseen_dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        shuffle=False,
        drop_last=True,
        persistent_workers=True
    )

    # Set up model
    model = setup_model(config).to(device)

    # Test Seen
    seen_metric = test_epoch(
        config=config,
        model=model,
        dataloader=seen_loader,
        device=device
    )

    # Test Unseen
    unseen_metric = test_epoch(
        config=config,
        model=model,
        dataloader=unseen_loader,
        device=device
    )

    metric = {}
    metric["test_seen"] = seen_metric
    metric["test_unseen"] = unseen_metric

    res_path = os.path.join(config["project_folder"], "result.json")
    with open(res_path, "w", encoding="utf-8") as json_file:
        json.dump(metric, json_file, ensure_ascii=False, indent=4)


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="UrbanNav")
    parser.add_argument("--project", "-p", type=str, help="Path to the project folder")
    parser.add_argument("--checkpoint", "-m", type=str, help="Path to the checkpoint")
    parser.add_argument("--gpu", "-g", type=int, help="Path to the checkpoint")
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--seen_split", type=str)
    parser.add_argument("--unseen_split", type=str)
    args = parser.parse_args()

    # Load config
    with open(os.path.join(args.project, 'train_config.json'), "r") as f:
        config = json.load(f)
    config.pop('project_name')
    config.pop('run_name')
    config["project_folder"] = args.project
    config["model_path"] = args.checkpoint
    config["mode"] = "test"
    config["batch_size"] = args.batch_size
    config["seen_split"] = args.seen_split
    config["unseen_split"] = args.unseen_split
    
    # Set random seed
    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    # Setup cuda
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"])
    device = torch.device(f"cuda:{args.gpu}")

    # Set train and test datasets
    _, _, test_seen_dataset, test_unseen_dataset = setup_dataset(config)

    # Test
    test(config, test_seen_dataset, test_unseen_dataset, device)

    