import matplotlib.pyplot as plt
from utils.data_loading import BasicDataset, CarvanaDataset, BasicAugmentedDataset
from utils.dice_score import *
from predict import predict_img
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from unet import UNet
import torchvision.transforms as T

from torch.utils.data import DataLoader, random_split



def hardening(array):
    print("hardening")
    for i in range(len(array)):
        for j in range(len(array[i])):
            for k in range(len(array[i][j])):
                if array[i][j][k] > 1:
                    array[i][j][k] = 1
                else: 
                    array[i][j][k] = 0
    return array

dir_img = Path('data/test_imgs')
dir_mask = Path('data/test_masks')

test_set = CarvanaDataset(dir_img, dir_mask)
print('test set successfully loaded with: ', len(test_set), " images\n")



net = UNet(n_channels=1, n_classes=3, bilinear=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

name_model = 'runs/Tracked runs/100epoche_restricted_3enc_MultistepLR_NEWfuzzy.pth'
print(f'Loading model {name_model}')
print(f'Using device {device}')

net.to(device=device)
state_dict = torch.load(name_model, map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)
print('Model loaded!')

net.eval()

val_loader = DataLoader(test_set, shuffle=False)

itr = val_loader._get_iterator()

masks = []
ress = []
for i in range(len(test_set)):

    data = itr._next_data()
    
    img = data['image'][0]
    
    mask = data['mask'][0]
    
    print(img.numpy().shape)
    if (mask.shape == torch.Size([144, 512])): # (== ROGU) (!= GULE)
        continue


    plt.imshow(T.ToPILImage()(img))
    plt.title("image " + str(i) + " of " + str(len(test_set)))
    plt.show()

    

