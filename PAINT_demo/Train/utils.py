import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader import *

# Initialization
def init_optimizer(all_parameter,lr,type):
    if type=='RMSprop':
        opt = torch.optim.RMSprop(all_parameter, lr=lr)
    elif type == 'Adam':
         opt = torch.optim.Adam(all_parameter, lr=lr)
    elif type == 'AdamW':
         opt = torch.optim.AdamW(all_parameter, lr=lr, weight_decay=0.0001)
    else:
        opt = torch.optim.SGD(all_parameter, lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300, eta_min=1e-8, last_epoch=-1)
    return opt,sch

def init_data(train_data_path,valid_data_path,batch_size):
    train_set = Load_Dataset(train_data_path)
    valid_set = Load_Dataset(valid_data_path)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=10)
    valid_loader = DataLoader(valid_set, batch_size = 1, shuffle=False, num_workers=10)
    return train_loader, valid_loader

def downsample(hsi):
    b, _, h, w = hsi.shape
    hr_msi = torch.zeros(b, 7, h, w).to("cuda")
    start_band = [0,0,8,20,43,105,143]
    end_band = [1,7,16,26,47,115,163]
    for i in range(7):
        hr_msi[:, i:i+1, :, :] = hsi[:, start_band[i]:end_band[i], :, :].mean(dim=1, keepdim=True)
    return hr_msi 

# Metric computation
def psnr(z_hat, z):
    MAX_VALUE = np.max(z**2, axis=(1,2))
    L = z_hat.shape[1] * z_hat.shape[2]
    error = np.sum((z_hat - z)**2, axis=(1,2))/L
    index = 10*np.log10(MAX_VALUE/error)
    m_idx = np.mean(index)
    return m_idx

def sam(z_hat, z):
    de_z_hat = np.sqrt(np.sum(z_hat**2, axis=0))
    de_z = np.sqrt(np.sum(z**2, axis=0))
    angle = np.rad2deg(np.arccos(np.sum(z_hat * z, axis=0)/(de_z_hat * de_z)))
    sam_idx = np.mean(angle, axis=(0,1))
    return sam_idx

def rmse(z_hat, z):
    L = z_hat.shape[1] * z_hat.shape[2]
    M = z_hat.shape[0]
    rmse_m = np.sqrt(np.sum((z_hat - z)**2, axis=(1, 2))) / np.sqrt(L)
    rmse = np.sqrt(np.sum(rmse_m**2)/M)
    return rmse

# Loss function
def spatial_tv_loss(x):
    h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return h_tv + w_tv

def spectral_tv_loss(x):
    c_tv = torch.abs(x[:, 1:, :, :] - x[:, :-1, :, :]).mean()
    return c_tv

def spectral_angle_mapper(preds, targets, eps=1e-8):
    preds = preds.permute(0, 2, 3, 1)  
    targets = targets.permute(0, 2, 3, 1)  
    preds_flat = preds.reshape(-1, preds.shape[-1]) 
    targets_flat = targets.reshape(-1, targets.shape[-1])
    dot_product = torch.sum(preds_flat * targets_flat, dim=1)
    preds_norm = torch.norm(preds_flat, dim=1)
    targets_norm = torch.norm(targets_flat, dim=1)
    cos = dot_product / (preds_norm * targets_norm + eps)
    cos = torch.clamp(cos, -1.0+eps, 1.0-eps)
    angles = torch.acos(cos)
    return torch.mean(angles)