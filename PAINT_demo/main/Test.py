import torch
import os
import scipy.io as sio
from network.PAINT import PAINT
from functions.utils import *
import os

current_directory = os.getcwd()
file_path = f"{current_directory}/main"
os.chdir(file_path)
device = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')    

def Test(test_path, test_loader):
    checkpoint = torch.load("checkpoint/" + "PAINT" + ".pt")
    Net = PAINT(spa_itr=4, spe_itr=3, Landsat_channels=7, AVIRIS_channels=172).to(device)
    Net.load_state_dict(checkpoint['Gnet'])
    for test_data in test_loader:
        Landsat, AVIRIS, Pan = test_data['Landsat'].to(device), test_data['AVIRIS'].to(device), test_data['Pan'].to(device)
        with torch.no_grad():
            _, _, model_out = Net(Landsat,Pan)
        y = model_out.squeeze().permute(1,2,0).cpu().numpy()
        Landsat = Landsat.squeeze().permute(1,2,0).cpu().numpy()
        AVIRIS = AVIRIS.squeeze().permute(1,2,0).cpu().numpy()  
        all_dict = {"Y_H": y, "AVIRIS": AVIRIS, "MSI": Landsat}
        sio.savemat(os.path.join('result/','PAINT_result.mat'), all_dict)                

if __name__ == "__main__" :
    test_path = "./data/"
    test_loader = init_data(test_path)
    Test(test_path=test_path, test_loader=test_loader)