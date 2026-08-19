# path 
save_path = "./PAINT"
train_data_path = "./dataset/Train_Spec/"
valid_data_path = "./dataset/Valid_Spec/"

# for training
num_workers = 10
batch_size  = 8
gpu_ids = 0
epoch_num = 800
val_period = 2
save_epoch = 1

# model settings
lr_Net = 1e-3
opt_Net = 'AdamW' 