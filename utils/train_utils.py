import os
import clip
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF



    
def mean_reduce(local_tensor: torch.tensor) -> torch.Tensor:
    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
    return local_tensor / dist.get_world_size()


def compute_loss(waypoint_pred, arrived_pred, feature_pred, waypoint_gt, arrived_gt, feature_gt):

    # Compute feature loss
    if feature_pred is not None and feature_gt is not None:
        feature_loss = F.mse_loss(feature_pred, feature_gt)
    else:
        feature_loss = 0.0

    # Compute Waypoiint loss
    waypoint_loss = F.l1_loss(waypoint_pred, waypoint_gt)

    # Compute arrived loss
    arrived_loss = F.binary_cross_entropy_with_logits(arrived_pred.flatten(), arrived_gt)

    # Compute direction loss
    waypoint_pred = waypoint_pred.view(-1, 2)
    waypoint_gt = waypoint_gt.view(-1, 2)

    dot_product = (waypoint_pred * waypoint_gt).sum(dim=1)
    waypoint_pred_norm = waypoint_pred.norm(dim=1)
    waypoint_gt_norm = waypoint_gt.norm(dim=1)
    cos_sim = dot_product / (waypoint_pred_norm * waypoint_gt_norm + 1e-8)

    direction_loss = 1 - cos_sim.mean()

    return waypoint_loss, arrived_loss, direction_loss, feature_loss


def train_epoch(
        config: dict,
        ddp_model: nn.Module,
        optimizer: Adam,
        dataloader: DataLoader,
        device: torch.device,
        epoch: int
):
    batch_size = config["batch_size"]
    context_size = config["context_size"]

    ddp_model.train()
    text_encoder, preprocess = clip.load("ViT-B/32")
    text_encoder = text_encoder.to(torch.float32).to(device)

    use_image = config["data"]["use_image"]

    direction_loss_weight = config["train"]["direction_loss_weight"]
    feature_loss_weight = config["train"]["feature_loss_weight"]
    arrived_loss_weight = config["train"]["arrived_loss_weight"]

    if device == 0:
        print(f"Start training for epoch {epoch}")
        training_bar = tqdm(dataloader, desc=f"Training for epoch {epoch}", leave=True, ncols=120)
    else:
        training_bar = dataloader

    epoch_losses = {}
    epoch_losses["loss"] = []
    epoch_losses["waypoint_loss"] = []
    epoch_losses["direction_loss"] = []
    epoch_losses["feature_loss"] = []
    epoch_losses["arrived_loss"] = []

    for i, data in enumerate(training_bar):

        instr = data["instr"]
        obs_feat = data["obs_feat"].to(device)
        feature_gt = data["future_feat"].to(device)
        input_poses = data["input_poses"].to(device)
        waypoint_gt = data["waypoint_poses"].to(device)
        arrived_gt = data["arrived"].to(device)

        if use_image:
            curr_frame = data["curr_frame"].to(device)
            curr_frame = TF.center_crop(curr_frame, config["model"]["crop"])
            curr_frame = TF.resize(curr_frame, config["model"]["resize"])

        # Extract text features
        with torch.no_grad():
            text_inputs = clip.tokenize(instr, truncate=True).to(device)
            text_feat = text_encoder.encode_text(text_inputs)

        # Predict
        waypoint_pred, arrived_pred, feature_pred = ddp_model(text_feat, curr_frame, obs_feat, input_poses)

        # Compute Loss
        waypoint_loss, arrived_loss, direction_loss, feature_loss = compute_loss(waypoint_pred, arrived_pred, feature_pred, waypoint_gt, arrived_gt, feature_gt)
        loss = waypoint_loss + arrived_loss_weight * arrived_loss + direction_loss_weight * direction_loss + feature_loss * feature_loss_weight
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # logging
        loss_cpu = mean_reduce(loss).item()
        if device == 0:
            training_bar.set_postfix(loss=loss_cpu)

        epoch_losses["loss"].append(loss_cpu)
        epoch_losses["waypoint_loss"].append(mean_reduce(waypoint_loss).item())
        epoch_losses["direction_loss"].append(mean_reduce(direction_loss).item())
        epoch_losses["feature_loss"].append(mean_reduce(feature_loss).item())
        epoch_losses["arrived_loss"].append(mean_reduce(arrived_loss).item())

    return epoch_losses


def val_epoch(
        config: dict,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        epoch: int
):
    
    batch_size = config["batch_size"]
    context_size = config["context_size"]
    use_image = config["data"]["use_image"]
    
    model.eval()
    text_encoder, preprocess = clip.load("ViT-B/32")
    text_encoder = text_encoder.to(torch.float32).to(device)

    direction_loss_weight = config["train"]["direction_loss_weight"]
    feature_loss_weight = config["train"]["feature_loss_weight"]
    arrived_loss_weight = config["train"]["arrived_loss_weight"]

    if device == 0:
        val_bar = tqdm(dataloader, desc=f"Evaluating for epoch {epoch}", leave=True, ncols=120)
    else:
        val_bar = dataloader

    epoch_losses = {}
    epoch_losses["loss"] = []
    epoch_losses["waypoint_loss"] = []
    epoch_losses["direction_loss"] = []
    epoch_losses["feature_loss"] = []
    epoch_losses["arrived_loss"] = []

    for i, data in enumerate(val_bar):

        instr = data["instr"]
        obs_feat = data["obs_feat"].to(device)
        feature_gt = data["future_feat"].to(device)
        input_poses = data["input_poses"].to(device)
        waypoint_gt = data["waypoint_poses"].to(device)
        arrived_gt = data["arrived"].to(device)

        if use_image:
            curr_frame = data["curr_frame"].to(device)
            curr_frame = TF.center_crop(curr_frame, config["model"]["crop"])
            curr_frame = TF.resize(curr_frame, config["model"]["resize"])

        # Extract text features
        with torch.no_grad():
            text_inputs = clip.tokenize(instr, truncate=True).to(device)
            text_feat = text_encoder.encode_text(text_inputs)
        
        # Predict
        with torch.no_grad():
            waypoint_pred, arrived_pred, feature_pred = model(text_feat, curr_frame, obs_feat, input_poses)

        # Compute Loss
        waypoint_loss, arrived_loss, direction_loss, feature_loss = compute_loss(waypoint_pred, arrived_pred, feature_pred, waypoint_gt, arrived_gt, feature_gt)
        loss = waypoint_loss + arrived_loss_weight * arrived_loss + direction_loss_weight * direction_loss + feature_loss * feature_loss_weight

        # Logging
        loss_cpu = mean_reduce(loss).item()
        if device == 0:
            val_bar.set_postfix(loss=loss_cpu)

        epoch_losses["loss"].append(loss_cpu)
        epoch_losses["waypoint_loss"].append(mean_reduce(waypoint_loss).item())
        epoch_losses["direction_loss"].append(mean_reduce(direction_loss).item())
        epoch_losses["feature_loss"].append(mean_reduce(feature_loss).item())
        epoch_losses["arrived_loss"].append(mean_reduce(arrived_loss).item())
        
    return epoch_losses


def test_epoch(
        config: dict,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        ):
    
    len_traj_pred = config["len_traj_pred"]
    use_image = config["data"]["use_image"]
    
    model.eval()
    text_encoder, preprocess = clip.load("ViT-B/32")
    text_encoder = text_encoder.to(torch.float32).to(device)

    direction_loss_weight = config["train"]["direction_loss_weight"]
    feature_loss_weight = config["train"]["feature_loss_weight"]

    test_bar = tqdm(dataloader, desc=f"Testing: ", leave=True, ncols=120)

    metric = {}
    metric["loss"] = []
    metric["waypoint_loss"] = []
    metric["direction_loss"] = []
    metric["feature_loss"] = []
    metric["arrived_loss"] = []
    metric["arrived_acc"] = []
    for i in range(len_traj_pred):
        metric[f"angle_{i+1}"] = []
        metric[f"distance_{i+1}"] = []
    metric["AOE"] = []
    metric["MAOE"] = []
    metric["ADE"] = []
    metric["MADE"] = []
    metric["weighted_fitness"] = []

    for i, data in enumerate(test_bar):

        instr = data["instr"]
        obs_feat = data["obs_feat"].to(device)
        feature_gt = data["future_feat"].to(device)
        input_poses = data["input_poses"].to(device)
        waypoint_gt = data["waypoint_poses"].to(device)
        arrived_gt = data["arrived"].to(device)

        if use_image:
            curr_frame = data["curr_frame"].to(device)
            curr_frame = TF.center_crop(curr_frame, config["model"]["crop"])
            curr_frame = TF.resize(curr_frame, config["model"]["resize"])

        # Extract text features
        with torch.no_grad():
            text_inputs = clip.tokenize(instr, truncate=True).to(device)
            text_feat = text_encoder.encode_text(text_inputs)
        
        # Predict
        with torch.no_grad():
            waypoint_pred, arrived_pred, feature_pred = model(text_feat, curr_frame, obs_feat, input_poses)

        # Compute loss and metric
        waypoint_loss, arrived_loss, direction_loss, feature_loss = compute_loss(waypoint_pred, arrived_pred, feature_pred, waypoint_gt, arrived_gt, feature_gt)
        loss = waypoint_loss + direction_loss_weight * direction_loss + feature_loss * feature_loss_weight
        waypoint_pred_view = waypoint_pred.view(-1, 2)
        waypoint_gt_view = waypoint_gt.view(-1, 2)

        # Compute arrived accuracy
        arrived_probs = torch.sigmoid(arrived_pred)
        arrived_pred_binary = (arrived_probs >= 0.5).float().squeeze(-1)
        arrived_correct = (arrived_pred_binary == arrived_gt).float()
        arrived_acc = arrived_correct.mean()

        # Compute AOE and MAOE
        cos_sim = F.cosine_similarity(waypoint_pred_view, waypoint_gt_view, dim=1)
        angles = torch.acos(cos_sim) * 180 / torch.pi
        angles = angles.view(config["batch_size"], config["len_traj_pred"])
        mean_angle_step = angles.mean(dim=0)
        mean_angle_step_np = mean_angle_step.cpu().numpy()
        max_angles, _ = torch.max(angles, dim=1)
        mean_angle_max = max_angles.mean(dim=0)
        mean_angle = torch.mean(angles)

        # Compute distance
        distances = torch.norm(waypoint_pred_view - waypoint_gt_view, p=2, dim=1)
        distances = distances.view(config["batch_size"], config["len_traj_pred"])
        mean_distance_step = distances.mean(dim=0)
        mean_distance_step_np = mean_distance_step.cpu().numpy()
        max_distances, _ = torch.max(distances, dim=1)
        mean_distance_max = max_distances.mean(dim=0)
        mean_distance = torch.mean(distances)

        # WF
        metric_factor = torch.linspace(1 / config['len_traj_pred'], 1, steps=config['len_traj_pred']).to(device)
        weighted_fitness = 0.1 * torch.dot(mean_angle_step, metric_factor.flip(0)) + torch.dot(mean_distance_step, metric_factor)

        metric["loss"].append(loss.item())
        metric["waypoint_loss"].append(waypoint_loss.item())
        metric["direction_loss"].append(direction_loss.item())
        metric["feature_loss"].append(feature_loss.item())
        metric["arrived_loss"].append(arrived_loss.item())
        
        for i in range(len_traj_pred):
            metric[f'angle_{i+1}'].append(float(mean_angle_step_np[i]))
            metric[f'distance_{i+1}'].append(float(mean_distance_step_np[i]))

        metric["AOE"].append(mean_angle.item())
        metric["MAOE"].append(mean_angle_max.item())
        metric["ADE"].append(mean_distance.item())
        metric["MADE"].append(mean_distance_max.item())
        metric["weighted_fitness"].append(weighted_fitness.item())
        metric["arrived_acc"].append(arrived_acc.item())

    for k, v in metric.items():
        metric[k] = np.nanmean(v)

    return metric

