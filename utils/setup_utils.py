import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
import torch.distributed as dist

import os
import random
import pickle as pkl
from data.urban_dataset import UrbanNavDataset
from model.urban_mlp import UrbanNavMLP
from model.urban_ca import UrbanNavCrossAttention
from model.urban_film import UrbanNav


def setup_DDP(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def setup_model(config: dict):
    
    if config["model"]["feature_fusion"] == "mlp":
        model = UrbanNavMLP(config)
    elif config["model"]["feature_fusion"] == "cross_attention":
        mode = UrbanNavCrossAttention(config)
    elif config["model"]["feature_fusion"] == "film":
        model = UrbanNav(config)
    else:
        raise NotImplementedError(f"Feature fusion method '{config['model']['feature_fusion']}' is not implemented.")

    if config["mode"] == "test":
        model.load_state_dict(torch.load(config["model_path"], weights_only=True))
    
    return model


def setup_optimizer(config: dict, model: nn.Module):

    lr = float(config["train"]["lr"])

    optimizer_type = config["train"]["optimizer"]
    if optimizer_type == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=config["train"]["weight_decay"])
    elif optimizer_type == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config["train"]["weight_decay"])
    elif optimizer_type == "sgd":
        optimizer = SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    scheduler_type = config["train"]["scheduler"]
    if scheduler_type == "step_lr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["train"]["step_size"], gamma=config["train"]["gamma"])
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown optimizer: {scheduler_type}")

    return optimizer, scheduler


# Set up dataset
def setup_dataset(config):

    train_dataset = None
    val_dataset = None
    test_seen_dataset = None
    test_unseen_dataset = None

    if config["mode"] in ['train', 'finetune']:

        with open(config["data"]["split_file"], 'rb') as split_file:
            data_list = pkl.load(split_file)

        random.shuffle(data_list)
        split_idx = int(len(data_list) * config["data"]["split"])
        train_data_list = data_list[:split_idx]
        val_data_list = data_list[split_idx:]
        
        train_dataset = UrbanNavDataset(config=config, mode="train", data_list=train_data_list)
        val_dataset = UrbanNavDataset(config=config, mode="val", data_list=val_data_list)

    elif config["mode"] == 'test':
        with open(config["seen_split"], 'rb') as seen_split_file:
            seen_data_list = pkl.load(seen_split_file)
        test_seen_dataset = UrbanNavDataset(config=config, mode="test", data_list=seen_data_list)

        with open(config["unseen_split"], 'rb') as unseen_split_file:
            unseen_data_list = pkl.load(unseen_split_file)
        test_unseen_dataset = UrbanNavDataset(config=config, mode="test", data_list=unseen_data_list)

    return train_dataset, val_dataset, test_seen_dataset, test_unseen_dataset

