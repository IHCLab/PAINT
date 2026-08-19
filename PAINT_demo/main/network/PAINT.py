import torch
import torch.nn.functional as F
import torch.nn as nn
from .function import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def b_matrix(x):
    x_scaled = F.avg_pool2d(x, kernel_size=2, stride=2)
    return x_scaled

class B_Operator(nn.Module):
    def __init__(self):
        super(B_Operator, self).__init__()    
        self.avg_pooling = nn.AvgPool2d(2)
    def forward(self, input_data):
        output = self.avg_pooling(input_data)
        return output

class B_Transpose_Operator(nn.Module):
    def __init__(self, in_channels=7):
        super(B_Transpose_Operator, self).__init__()   
        self.in_channels = in_channels
        self.hidden_layer = 64
        self.Conv_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, output_padding=1, groups=self.in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1, groups=self.in_channels),
        )
    def forward(self, input_data):
        output = self.Conv_block(input_data)
        return output

def spatial_update_rule(Z, V, Y_L, P, D, shared_rho, in_channels, out_channels, conv1x1_D, conv1x1_U, Proximal_operator, B_operator, B_T_operator):        
    batch_size, _, height, width = P.shape
    P_flat = P.reshape(batch_size, in_channels, -1)       
    D_transpose_P = torch.matmul(D.transpose(0, 1), P_flat) 
    D_transpose_P = D_transpose_P.reshape(batch_size, out_channels, height, width)  
    Y_L_B_transpose = B_T_operator(Y_L)    
    V = V_update_spatial(Z, V, D_transpose_P, shared_rho, conv1x1_D, conv1x1_U)  
    Z = Z_update_spatial(Z, V, Y_L_B_transpose, Proximal_operator, B_operator, B_T_operator, shared_rho)                 
    return Z, V

class SpatialSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=7, num_iterations=4):
        super(SpatialSR, self).__init__()     
        self.num_iterations, self.in_channels, self.out_channels = num_iterations, in_channels, out_channels   
        self.prox_operator = ProximalOperator(out_channels) 
        self.B_operator = B_Operator()
        self.B_T_operator = B_Transpose_Operator(in_channels=self.out_channels)              
        self.shared_conv1x1_D = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)
        self.shared_conv1x1_U = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.shared_rho = nn.Parameter(torch.ones(1, device=device))
        self.shared_D = nn.Parameter(torch.randn(in_channels, out_channels))        
        nn.init.xavier_normal_(self.shared_D)
    def forward(self, Y_L, P):
        for stage in range(self.num_iterations):
            if stage==0:
                Z, V = initial_model(Y_L, P)   
            Z, V = spatial_update_rule(Z=Z, V=V, Y_L=Y_L, P=P, D=self.shared_D, shared_rho=self.shared_rho, in_channels=self.in_channels, 
                                       out_channels=self.out_channels, conv1x1_D=self.shared_conv1x1_D, conv1x1_U=self.shared_conv1x1_U,
                                       Proximal_operator=self.prox_operator, B_operator=self.B_operator, B_T_operator=self.B_T_operator)
        return Z

def spectral_update_rule(Y_L, Yh, U, D, in_channels, out_channels, shared_rho, conv1x1_D, conv1x1_U, channelFc, prox_operator):
    batch_size, _, height, width = Y_L.shape
    Ys_flat = Y_L.reshape(batch_size, in_channels, -1)         
    D_transpose_Y_L = torch.matmul(D.transpose(0, 1), Ys_flat)
    D_transpose_Y_L = D_transpose_Y_L.reshape(batch_size, out_channels, height, width)
    Z = Z_update_spectral(Yh, U, prox_operator)        
    Yh = Yh_update_spectral(Z, U, D_transpose_Y_L, shared_rho, conv1x1_D, conv1x1_U, channelFc)        
    U = U_update_spectral(U, Yh, Z)         
    return Z, Yh, U

class SpectralSR(nn.Module):
    def __init__(self, in_channels=7, out_channels=86, num_iterations=4):
        super(SpectralSR, self).__init__()
        self.num_iterations, self.in_channels, self.out_channels = num_iterations, in_channels, out_channels
        self.shared_conv1x1_D = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)
        self.shared_conv1x1_U = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.shared_D = nn.Parameter(torch.randn(in_channels, out_channels))  
        self.shared_rho = nn.Parameter(torch.ones(1, device=device))      
        self.shared_prox_operator = ProximalOperator(out_channels)
        self.shared_channel_fc = SymmetricLinear(in_channels)
        nn.init.xavier_normal_(self.shared_D)
        self.initial_estimate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1), 
            nn.ReLU(inplace=True)
        )
    def forward(self, Y_L):
        for stage in range(self.num_iterations):
            if stage==0:
                Yh = self.initial_estimate(Y_L)   
                U = torch.zeros_like(Yh)
                Z, Yh, U = spectral_update_rule(Y_L=Y_L, Yh=Yh, U=U, D=self.shared_D, in_channels=self.in_channels, out_channels=self.out_channels,
                                                shared_rho=self.shared_rho, conv1x1_D=self.shared_conv1x1_D, conv1x1_U=self.shared_conv1x1_U,
                                                channelFc=self.shared_channel_fc, prox_operator=self.shared_prox_operator)
            elif stage==self.num_iterations-1:
                Z = Z_update_spectral(Yh, U, self.shared_prox_operator)
            else:
                Z, Yh, U = spectral_update_rule(Y_L=Y_L, Yh=Yh, U=U, D=self.shared_D, in_channels=self.in_channels, out_channels=self.out_channels,
                                                shared_rho=self.shared_rho, conv1x1_D=self.shared_conv1x1_D, conv1x1_U=self.shared_conv1x1_U,
                                                channelFc=self.shared_channel_fc, prox_operator=self.shared_prox_operator)
        return Z
    
class PAINT(nn.Module):
    def __init__(self, spa_itr, spe_itr, Landsat_channels=7, AVIRIS_channels=172):
        super(PAINT, self).__init__()
        self.spatial_SR = SpatialSR(num_iterations=spa_itr)      
        self.spectral_SR = SpectralSR(in_channels=Landsat_channels, out_channels=AVIRIS_channels//2, num_iterations=spe_itr)   
        self.upsample_block = nn.Sequential(
            nn.Conv2d(in_channels=AVIRIS_channels//2, out_channels=AVIRIS_channels, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=AVIRIS_channels, out_channels=AVIRIS_channels, kernel_size=3, padding=1, groups=4),
        )
    def forward(self, Landsat_data, Pan_data):
        PGD_output = self.spatial_SR(Landsat_data, Pan_data)
        ADMM_output = self.spectral_SR(PGD_output)
        SCPM_output = self.upsample_block(ADMM_output)    
        return PGD_output, ADMM_output, SCPM_output        