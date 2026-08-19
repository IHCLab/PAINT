import torch
import torch.nn as nn
import numpy as np
import config as cfg
import warnings
from os.path import join
from tqdm import tqdm
from architecture.PAINT import PAINT
from utils import *
warnings.filterwarnings("ignore")
import os
current_directory = os.getcwd()
file_path = f"{current_directory}/Train"
os.chdir(file_path)

class Trainer(nn.Module):
    def __init__(self):
        super().__init__()   
        self.device = torch.device('cuda:{}'.format(cfg.gpu_ids)) if torch.cuda.is_available() else torch.device('cpu')
        self.train_loader, self.val_loader = init_data(cfg.train_data_path,cfg.valid_data_path,cfg.batch_size)
        self.Net = PAINT(spa_itr=4, spe_itr=3, Landsat_channels=7, AVIRIS_channels=172).to(self.device)
        self.opt,self.sch = init_optimizer(self.Net.parameters(), lr = cfg.lr_Net, type = cfg.opt_Net)
        self.loss = nn.L1Loss()
    def optimize(self,train_data):
        Landsat = train_data['Landsat'].to(self.device)
        AVIRIS = train_data['AVIRIS'].to(self.device)
        Pan = train_data['Pan'].to(self.device)
        AVIRIS,Landsat,Pan = self.crop(AVIRIS,Landsat,Pan)
        self.Net.train()
        self.opt.zero_grad()
        spatial_output, intermediate_output, final_output = self.Net(Landsat,Pan)
        hr_msi = downsample(AVIRIS)
        err = 1e-5*self.loss(intermediate_output, AVIRIS[:, 1::2, :, :])+self.loss(final_output, AVIRIS)+0.2*spectral_angle_mapper(final_output, AVIRIS)+self.loss(spatial_output, hr_msi)+0.001*spectral_tv_loss(final_output)+1e-8*spatial_tv_loss(final_output) 
        err.backward() 
        self.opt.step()       
        return err
    def crop(self, data, data2, data3, crop_size=(64,64)):
        _,_, w, h = data.shape
        w_patch, h_patch =crop_size
        w_idx = np.random.randint(0, w - w_patch)
        h_idx = np.random.randint(0, h - h_patch)
        crop_patch = data[:,:, w_idx:w_idx + w_patch, h_idx:h_idx + h_patch]
        crop_patch2 = data2[:,:, w_idx:w_idx + w_patch, h_idx:h_idx + h_patch]
        crop_patch3 = data3[:,:, w_idx:w_idx + w_patch, h_idx:h_idx + h_patch]
        return crop_patch,crop_patch2,crop_patch3
    def train_part(self, train_loader):      
        train_loss_list = [] 
        for train_data in train_loader:     
            err = self.optimize(train_data)
            train_loss_list.append(err.item())     
        self.sch.step()
        train_loss = np.array(train_loss_list).mean()
        return train_loss
    def eval(self,dataset,epoch):
        self.Net.eval()
        PSNR_list, SAM_list = [], []
        for val_data in dataset:
            Landsat = val_data['Landsat'].to(self.device)
            AVIRIS = val_data['AVIRIS'].to(self.device)
            Pan = val_data['Pan'].to(self.device)
            with torch.no_grad():
                _,_,model_out  = self.Net(Landsat,Pan)
            y = model_out.squeeze().permute(1,2,0).cpu().numpy()
            AVIRIS = AVIRIS.squeeze().permute(1,2,0).cpu().numpy()
            PSNR = psnr(AVIRIS.transpose(2,0,1),y.transpose(2,0,1))
            SAM = sam(AVIRIS.transpose(2,0,1),y.transpose(2,0,1))
            PSNR_list.append(PSNR)
            SAM_list.append(SAM)
        PSNR_avg = np.array(PSNR_list).mean()
        SAM_avg = np.array(SAM_list).mean()
        msg = "[Epoch_{}] PSNR (HSI): {} SAM (HSI): {}".format(epoch, PSNR_avg, SAM_avg)
        print(msg) 
    def forward(self):
        for i in tqdm(range(cfg.epoch_num)):
            train_loss = self.train_part(self.train_loader)
            msg = "[Epoch_{}] Training_Loss: {} " .format(i, train_loss)
            print(msg)
            if i % cfg.save_epoch==0:
                torch.save({'Gnet':self.Net.state_dict()}, join(cfg.save_path, "model_epoch_{}.pt".format(i)))                  
            if i % cfg.val_period==0:
                print('\n Testing')
                self.eval(self.val_loader,i)
                print('-' * 50)  
            with open(join(cfg.save_path, "msg_files"), "a+") as f:
                f.write(msg + '\n')
                f.write(str(train_loss) + '\n')
            
if __name__ == "__main__" :
    model = Trainer()
    model()