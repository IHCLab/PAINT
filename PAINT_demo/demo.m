%===========================================================================================
% Input:
% The 7-band Landsat-8/9 multispectral image and the panchromatic band are the input data, 
% whose dimension are 256*256*7 and 256*256*1, respectively.
%-------------------------------------------------------------------------------------------
% Output:
% Y_H represents the 172-band hyperspectral reconstruction result of PAINT, 
% whose dimension is 256*256*172.
%===========================================================================================
close all; clear; clc;
addpath(genpath('./main')); 
%% Setting
num_random_pixels = 5; height = 256; width = 256;
xy_list = [126,193;51,32;25,190;223,16;130,125];
rows = xy_list(:, 2); cols = xy_list(:, 1);
random_indices = sub2ind([256, 256], rows, cols);
%% Derivation of AVIRIS-level HSI from Landsat-8/9 by PAINT
tic
system(sprintf('conda run -n env_name python main/Test.py')); % env_name: the name of the Python environment you created
toc
%% Visualization
load(fullfile('./main/result', ['PAINT_result' '.mat']));
plot_result(AVIRIS, Y_H, MSI, random_indices, num_random_pixels, cols, rows)