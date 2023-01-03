import sfm
import torch
from scipy.optimize import minimize
import loader
import yaml
import tqdm
from pathlib import Path
import numpy as np


def read_params_path(params_path: Path):
    if params_path is None:
        raise ValueError("A parameters file must be specified for 'global' initialization mode.")
    elif not params_path.exists():
        raise ValueError(f"{params_path} is an incorrect parameters path.")
    with open(params_path, 'r') as f:
        params = yaml.load(f, yaml.Loader)
    return params


def estimate_global_parameters(image_list: list[sfm.Image], output_dir: Path):
    output_path = output_dir / 'global.yml'
    if output_path.exists():
        print(f'{output_path} already exists. Using it as initial parameters.')
        return

    print('Estimate veiling light and select pixels for dark/bright channel priors.')
    dcp = [
        {'Ic': [], 'z': []},
        {'Ic': [], 'z': []},
        {'Ic': [], 'z': []}
    ]
    bcp = [
        {'Ic': [], 'z': []},
        {'Ic': [], 'z': []},
        {'Ic': [], 'z': []}
    ]
    n_obs = 0
    B = torch.zeros(3, dtype=torch.float32, device='cuda')
    for image_idx, image_image, image_depth in tqdm.tqdm(
            loader.load_images(image_list, num_workers=10)):
        image = image_list[image_idx]
        I = image_image.to('cuda')
        z = image.distance_map(image_depth.to('cuda'))
        args_valid = z > 0
        n_obs += torch.sum(~args_valid)
        B += I[~args_valid].sum(dim=0)
        I_valid, z_valid = I[args_valid], z[args_valid]
        z_bounds = np.linspace(z_valid.min().item(), z_valid.max().item(), 11)
        for z_min, z_max in zip(z_bounds[:-1], z_bounds[1:]):
            args_range = (z_valid >= z_min) & (z_valid < z_max)
            if args_range.sum() > 0:
                for channel in range(3):
                    Ic_range = I_valid[:, channel][args_range]
                    z_range = z_valid[args_range]
                    args_dark = Ic_range < np.percentile(Ic_range.cpu(), 100 / 1024)
                    dcp[channel]['Ic'].append(Ic_range[args_dark].cpu())
                    dcp[channel]['z'].append(z_range[args_dark].cpu())
                    args_bright = Ic_range > np.percentile(Ic_range.cpu(), 100 * 1023 / 1024)
                    bcp[channel]['Ic'].append(Ic_range[args_bright].cpu())
                    bcp[channel]['z'].append(z_range[args_bright].cpu())
    B = B / n_obs
    print(f'B: {B.tolist()}')

    print('Estimate gamma width dark channel prior.')
    gamma = torch.zeros(3, dtype=torch.float32, device='cuda')
    for channel in range(3):
        Bc = B[channel]
        Ic = torch.hstack(dcp[channel]['Ic']).to('cuda')
        z = torch.hstack(dcp[channel]['z']).to('cuda')
        args_valid = Ic < Bc
        Ic_valid, z_valid = Ic[args_valid], z[args_valid]
        gammac_init = (torch.log(Bc / (Bc - Ic_valid)) * z_valid).sum() / z_valid.square().sum()

        def residuals(x):
            return torch.square(Ic - Bc * (1 - torch.exp(-x[0] * z))).sum().item()

        gamma[channel] = minimize(
            residuals,
            gammac_init.item(),
            method='Nelder-Mead',
            options={'disp': True}
        ).x[0]
    print(f'gamma: {gamma.tolist()}')

    print('Estimate beta and Jmax with bright channel prior.')
    beta = torch.zeros(3, dtype=torch.float32, device='cuda')
    Jmax = torch.zeros(3, dtype=torch.float32, device='cuda')
    for channel in range(3):
        Ic = torch.hstack(bcp[channel]['Ic']).to('cuda')
        z = torch.hstack(bcp[channel]['z']).to('cuda')
        Dc = Ic - B[channel] * (1 - torch.exp(-gamma[channel] * z))
        args_valid = Dc > 0
        Dc_valid, z_valid = Dc[args_valid], z[args_valid]
        Nc = Dc_valid.log()
        Xc = z_valid.sum() / z_valid.square().sum()
        Yc = (Nc * z_valid).sum() / z_valid.square().sum()
        Qc = 1 - Xc * z_valid
        Mc = (Qc * (Nc - Yc * z_valid)).sum() / Qc.square().sum()
        betac_init = Mc * Xc - Yc
        Jcmax_init = Mc.exp()

        def residuals(x):
            betac_hat, Jcmax_hat = x.tolist()
            return torch.square(Dc - Jcmax_hat * torch.exp(-betac_hat * z)).sum().item()

        beta[channel], Jmax[channel] = minimize(
            residuals,
            np.array([betac_init.item(), Jcmax_init.item()]),
            method='Nelder-Mead',
            options={'disp': True}
        ).x.tolist()
    print(f'beta: {beta.tolist()}\nJmax: {Jmax.tolist()}')

    print('Find Jmin with dark channel prior using all parameters.')
    Jmin = torch.zeros(3, dtype=torch.float32, device='cuda')
    for channel in range(3):
        Ic = torch.hstack(dcp[channel]['Ic']).to('cuda')
        z = torch.hstack(dcp[channel]['z']).to('cuda')
        Jmin[channel] = torch.median(
            (Ic - B[channel] * (1 - torch.exp(-gamma[channel] * z))) * torch.exp(beta[channel] * z)
        )
    print(f'Jmin: {Jmin.tolist()}')
    
    with open(output_path, 'w') as yaml_file:
        yaml.dump(
            {
                'B': B.tolist(),
                'beta': beta.tolist(),
                'gamma': gamma.tolist(),
                'Jmin': Jmin.tolist(),
                'Jmax': Jmax.tolist()
            },
            yaml_file
        )
