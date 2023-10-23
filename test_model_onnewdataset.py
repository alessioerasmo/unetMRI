import matplotlib.pyplot as plt
from utils.data_loading import BasicDataset, CarvanaDataset, BasicAugmentedDataset
from utils.dice_score import *
from utils.test_loading import TestDataset
from predict import predict_img
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from unet import UNet
import torchvision.transforms as T




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

test_set = TestDataset(dir_img, dir_mask)
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

masks = []
ress = []
for i in range(len(test_set)):
    
    img = test_set[i]['image']
    
    mask = test_set[i]['mask']
    if (mask.shape == torch.Size([144, 512])): # (== ROGU) (!= GULE)
        continue
    
    res = predict_img(net, T.ToPILImage()(img), device, 1)

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

values = []
for i in range(len(ress)):
    for j in range(len(ress[i])):
        for k in range(len(ress[i][j])):
            if ress[i][j][k] not in values:
                values.append(ress[i][j][k])
print("\n\n- prediction values in ", values)

values = []
for i in range(len(masks)):
    for j in range(len(masks[i])):
        for k in range(len(masks[i][j])):
            if masks[i][j][k] not in values:
                values.append(masks[i][j][k])
print("- mask values in ", values,"\n\n")

np_masks = np.array(masks, dtype='float32')
#np_result = np.array(ress, dtype='float32')
np_result = np.array(hardening(ress), dtype='float32')
#np_result = np.array(hardening_max(ress))

assert np_masks.shape == np_result.shape

for i in range(len(np_masks)):
    continue
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(np_masks[i])
    axarr[1].imshow(np_result[i]*255)
    #axarr[2].imshow(np.array(image)-(np.array(mask)*255))
    plt.show()


masks = torch.tensor(np_masks)
ress = torch.tensor(np_result)

dice = dice_coeff(ress, masks)

print(masks.shape, " ",ress.shape)

print(dice)


import nibabel as nib


GULEaffine = [[-1.09967160e+00, -1.21679362e-02, -5.98291550e-09,  7.91900711e+01],
              [ 2.68744361e-02, -4.97898221e-01, -5.83848747e-09,  1.09658463e+02],
              [-1.28950433e-08, -5.98291505e-09,  4.98046875e-01, -1.33319611e+02],
              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
ROGUaffine = [[-6.97692811e-01, -5.33618592e-02, -2.84517370e-02,  1.08155357e+02],
             [ 5.54330721e-02, -7.16394126e-01, -1.92132026e-01,  1.52908997e+02],
             [-1.28090577e-02, -1.93749443e-01,  7.18249738e-01, -1.18026794e+02],
             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]


#nii_mask = nib.Nifti1Image(np_masks, affine=GULEaffine)
nii_ress = nib.Nifti1Image(np_result, affine=ROGUaffine)

#nib.save(nii_mask, "ConsensusGULE.nii.gz")
nib.save(nii_ress, "FuzzyClassifierROGU.nii.gz")

"""
from scipy.io import savemat

#savemat("ConsensusROGU", {'matlab_array': np_masks})
#savemat("FuzzyClassifierGULE", {'matlab_array': ress})
"""


 