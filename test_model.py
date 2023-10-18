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

dir_img = Path('data/test_imgs')
dir_mask = Path('data/test_masks')

test_set = CarvanaDataset(dir_img, dir_mask)
print('test set successfully loaded with: ', len(test_set), " images\n")


net = UNet(n_channels=1, n_classes=2, bilinear=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

name_model = 'runs/Tracked runs/150epoche_restricted_3enc_MultistepLR_4xaffine_crisp.pth'
print(f'Loading model {name_model}')
print(f'Using device {device}')

net.to(device=device)
state_dict = torch.load(name_model, map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)
print('Model loaded!')

net.eval()

masks = []
ress = []
for i in range(len(test_set)):
    
    img = test_set[i]['image']
    
    mask = test_set[i]['mask']
    if (mask.shape == torch.Size([144, 512])):
        continue
    
    res = predict_img(net, T.ToPILImage()(img), device, 1)/7

    masks.append(mask.numpy())
    ress.append(res)

    dice = dice_coeff(torch.tensor(res), mask)
    plot = False
    if dice >= 0.01 and plot:
        print("dice: " +  str(dice))
        f, axarr = plt.subplots(1,3)
        axarr[0].imshow(T.ToPILImage()(img))
        axarr[1].imshow(mask)
        axarr[2].imshow(res)
        plt.show()


masks = torch.tensor(np.array(masks))
ress = torch.tensor(np.array(ress))

dice = dice_coeff(ress, masks)

print(masks.shape, " ",ress.shape)

print(dice)
