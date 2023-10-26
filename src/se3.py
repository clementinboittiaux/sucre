import torch
from torch import Tensor


def exp(pose: Tensor) -> tuple[Tensor, Tensor]:
    w1, w2, w3, p1, p2, p3 = pose
    zero = torch.zeros_like(w1)
    pose = torch.stack([zero, -w3, w2, p1, w3, zero, -w1, p2, -w2, w1, zero, p3, zero, zero, zero, zero])
    pose = torch.matrix_exp(pose.view(4, 4))
    return pose[:3, :3], pose[:3, 3:4]
