import torch.nn.functional as F
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initial_model(msi, pan):
    msi_scaled = F.interpolate(msi, scale_factor=2, mode='nearest')
    fused_image = msi_scaled+0.1*pan
    return fused_image, fused_image

def b_transpose_matrix(x):
    x = x * 0.25
    x_scaled = F.interpolate(x, scale_factor=2, mode='nearest')
    return x_scaled

def b_matrix(x):
    x_scaled = F.avg_pool2d(x, kernel_size=2, stride=2)
    return x_scaled

def Z_update_spectral(Yh, U, prox_operator):
    Z = prox_operator(Yh - U)
    return Z

def Yh_update_spectral(Z, U, D_transpose_Y_L, shared_rho, conv1x1_D, conv1x1_U, channelFc):        
    rhs = 2 * D_transpose_Y_L + shared_rho * (Z + U) 
    A = conv1x1_D(rhs)     
    batch_size, channels, height, width = A.shape
    A_flat = A.permute(0, 2, 3, 1).reshape(-1, channels)
    Phi = channelFc(A_flat)
    N = Phi.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
    B = conv1x1_U(N)         
    Yh = (rhs - (2 / shared_rho) * B)/shared_rho
    return Yh

def Yh_update_spectral_Woodbury_ablation(Z, U, D_transpose_Y_L, shared_rho, conv1x1_D, conv1x1_U, channelFc):        
    rhs = 2 * D_transpose_Y_L + shared_rho * (Z + U) 
    A = conv1x1_D(rhs)     
    batch_size, channels, height, width = A.shape
    A_flat = A.permute(0, 2, 3, 1).reshape(-1, channels)
    Phi = channelFc(A_flat)
    N = Phi.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
    B = conv1x1_U(A)         
    Yh = (rhs - (2 / shared_rho) * B)/shared_rho
    return Yh

def U_update_spectral(U, Yh, Z):
    return U - Yh + Z

def V_update_spatial(Z, V, D_transpose_P, shared_rho, conv1x1_D, conv1x1_U):  
    beta = 0.001      
    rhs = 2*conv1x1_U(conv1x1_D(V))+shared_rho*(V-Z)-2*D_transpose_P      
    V = V+beta*rhs
    return V

def Z_update_spatial(Z, V, Y_L_B_transpose, Proximal_operator, B_operator, B_T_operator, shared_rho):
    beta = 0.001
    rhs = 2*B_T_operator(B_operator(Z))+shared_rho*(Z-V)-2*Y_L_B_transpose
    Z_temp = Z+beta*rhs
    Z = Proximal_operator(Z_temp)   
    return Z

class SymmetricLinear(nn.Module):
    def __init__(self, size):
        super(SymmetricLinear, self).__init__()
        self.size = size
        self.lower_triangular = nn.Parameter(torch.randn(size, size))
        self.bias_param = nn.Parameter(torch.zeros(size))
    def forward(self, x):
        W = torch.tril(self.lower_triangular) + torch.tril(self.lower_triangular, -1).T
        out = x @ W.T
        out = out + self.bias_param
        return out

class ResBlock(nn.Module):
    def __init__(self, n_channels, kernel_size):
        super(ResBlock, self).__init__()
        if n_channels % 4 == 0:
            self.groups = 4
        else:
            self.groups = 1
        self.resblock = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size, stride=1, padding=kernel_size //2, bias=True, groups=self.groups), 
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, kernel_size, stride=1, padding=kernel_size //2, bias=True, groups=self.groups), 
        )
        self.relu = nn.ReLU()
    def forward(self,x):
        res = self.resblock(x)
        x = res + x
        return self.relu(x)
    
class ProximalOperator(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(ProximalOperator, self).__init__()
        if channels % 4 == 0:
            self.groups = 4
        else:
            self.groups = 1
        self.prox_network = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=True, groups=self.groups), 
            nn.ReLU(inplace=True),
            ResBlock(channels, kernel_size),
            ResBlock(channels, kernel_size),
            ResBlock(channels, kernel_size),
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size//2, bias=True, groups=self.groups) 
        )      
    def forward(self, x):
        prox = self.prox_network(x)
        x = prox + x
        return x