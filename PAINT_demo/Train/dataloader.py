import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join
from os import listdir
from scipy.io import loadmat

class Load_Dataset(Dataset):
    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._fname    = [join(root_dir, i) for i in listdir(root_dir) if i.endswith(".mat")]
    def __len__(self):
        return len(self._fname)  
    def __getitem__(self, idx):
        Landsat, AVIRIS, Pan = self._lmat(self._fname[idx])
        Landsat = torch.from_numpy(Landsat.astype(np.float32)).permute(2,0,1).type(torch.FloatTensor)
        AVIRIS = torch.from_numpy(AVIRIS.astype(np.float32)).permute(2,0,1).type(torch.FloatTensor)
        Pan = torch.from_numpy(Pan.astype(np.float32)).permute(2,0,1).type(torch.FloatTensor)
        all_dict = {"Landsat": Landsat, "AVIRIS":AVIRIS, "Pan":Pan, "fname": self._fname[idx],}
        return all_dict
    def _lmat(self, fn):
        x = loadmat(fn)
        return x['Y'],x['X'],x['Pan']