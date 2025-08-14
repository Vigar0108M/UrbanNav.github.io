import os
import yaml
import numpy as np

import torch
from torchvision import transforms


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()


def transform_images(images: torch.Tensor, transform: transforms):

    images_list = list(torch.split(images, 3, dim=1))
    for i in range(len(images_list)):
        images_list[i] = transform(images_list[i])
    
    return torch.concat(images_list, dim=1)


def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta


def normalize_data(data, stats):
    # nomalize to [0, 1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


def get_action(diffusion_output, action_stats):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    device = diffusion_output.device
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def action_reduce(unreduced_loss: torch.Tensor, action_mask: torch.Tensor):
    # Reduce over non-batch dimensions to get loss per batch element
    while unreduced_loss.dim() > 1:
        unreduced_loss = unreduced_loss.mean(dim=-1)
    assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
    return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)


def sinc_apx(angle):
    return torch.sin(3.141592*angle + 0.000000001)/(3.141592*angle + 0.000000001)


def twist_to_pose_diff_torch(v, w, dt):
    """
    Integrate 2D twist to get pose difference.
    Assuming constant velocity during time period `dt`.
    Args:
        v (float): velocity
        w (float): angular velocity
        dt (float): time delta

    """
    theta = -w  * dt
    z = v * dt * sinc_apx(-theta / np.pi)
    x = -v * dt * sinc_apx(-theta / (2 * np.pi)) * torch.sin(-theta / 2)
    return x, z, theta


def robot_pos_model_fix(linear_vel, angular_vel):
    # velocity commands integral
    bs, chorizon = linear_vel.shape
    device = linear_vel.device

    px = []
    pz = []
    pyaw = []
    Tacc = torch.eye(4, 4).unsqueeze(0).repeat(bs,1,1).to(device)
    for i in range(chorizon):
        x, z, yaw = twist_to_pose_diff_torch(linear_vel[:, i], angular_vel[:, i], 0.333)
        Todom = torch.zeros((bs, 4, 4)).to(device)
        Todom[:, 0, 0] = torch.cos(yaw)
        Todom[:, 0, 2] = torch.sin(yaw)
        Todom[:, 1, 1] = 1.0
        Todom[:, 2, 0] = -torch.sin(yaw)
        Todom[:, 2, 2] = torch.cos(yaw)
        Todom[:, 0, 3] = x
        Todom[:, 2, 3] = z
        Todom[:, 3, 3] = 1.0        
        
        Tacc = torch.matmul(Tacc, Todom)
               
        pyaw.append(torch.arctan(Tacc[:, 0, 2]/(Tacc[:, 0, 0] + 0.000000001)))        
        px.append(Tacc[:, 0, 3])
        pz.append(Tacc[:, 2, 3])        
    return px, pz, pyaw

