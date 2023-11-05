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
import os 




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


subject = "GULE"
classifier = "Binary"
epoche = 30

for subject in ["GULE", "ROGU"]:
    for classifier in ["Binary", "Fuzzy"]:
        for epoche in [30, 60, 100]:
            
            print("\n\nProcessing ", subject, " subject on " + classifier + " model trained on " + str(epoche)+ " epochs:")
            path_to_save = "niftiSAME_"+ str(epoche)
            
            if not os.path.exists(path_to_save):  
                os.makedirs(path_to_save) 
            
            dir_img = Path('data/test_imgs/'+subject)
            dir_mask = Path('data/test_masks/'+subject)

            test_set = TestDataset(dir_img, dir_mask)
            print('\t\ttest set successfully loaded with: ', len(test_set), " images\n")


            if (classifier=="Fuzzy"):
                net = UNet(n_channels=1, n_classes=3, bilinear=False)
            else:
                net = UNet(n_channels=1, n_classes=2, bilinear=False)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            models = {
                "Binary":{
                    30: 'runs/Tracked runs/30epoche_restricted_3enc_MultistepLR_crisp.pth',
                    60: 'runs/Tracked runs/60epoche_restricted_3enc_MultistepLR_crisp.pth',
                    100: 'runs/Tracked runs/100epoche_restricted_3enc_MultistepLR_crisp.pth'
                },
                "Fuzzy": {
                    30: 'runs/Tracked runs/30epoche_restricted_3enc_MultistepLR_NEWfuzzy.pth',
                    60: 'runs/Tracked runs/60epoche_restricted_3enc_MultistepLR_NEWfuzzy.pth',
                    100: 'runs/Tracked runs/100epoche_restricted_3enc_MultistepLR_NEWfuzzy.pth'
                }
            }
            name_model = models[classifier][epoche]

            print(f'\t\tLoading model {name_model}')
            print(f'\t\tUsing device {device}')

            net.to(device=device)
            state_dict = torch.load(name_model, map_location=device)
            mask_values = state_dict.pop('mask_values', [0, 1])
            net.load_state_dict(state_dict)
            print('\t\tModel loaded!')

            net.eval()

            masks = []
            ress = []
            for i in range(len(test_set)):
                
                img = test_set[i]['image']
                mask = test_set[i]['mask']
                res = predict_img(net, T.ToPILImage()(img), device, 1)

                masks.append(mask.numpy())
                ress.append(res)

            values = []
            for i in range(len(ress)):
                for j in range(len(ress[i])):
                    for k in range(len(ress[i][j])):
                        if ress[i][j][k] not in values:
                            values.append(ress[i][j][k])
            print("\t\t\t\t- prediction values in ", values)

            values = []
            for i in range(len(masks)):
                for j in range(len(masks[i])):
                    for k in range(len(masks[i][j])):
                        if masks[i][j][k] not in values:
                            values.append(masks[i][j][k])
            print("\t\t\t\t- mask values in ", values,"\n\n")

            np_masks = np.array(masks, dtype='float32')

            if (classifier == "Fuzzy"):
                np_result = np.array(hardening(ress), dtype='float32')
            else:
                np_result = np.array(ress, dtype='float32')

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

            print("\t\t\t\t",masks.shape, " ",ress.shape)

            print("\t\t\t\t",dice, " for ", subject, " classified by " + classifier + " trained on " + str(epoche)+ " epochs")


            import nibabel as nib

            affine = {
                "GULE": [[-1.09967160e+00, -1.21679362e-02, -5.98291550e-09,  7.91900711e+01],
                        [ 2.68744361e-02, -4.97898221e-01, -5.83848747e-09,  1.09658463e+02],
                        [-1.28950433e-08, -5.98291505e-09,  4.98046875e-01, -1.33319611e+02],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]],
                "ROGU": [[-6.97692811e-01, -5.33618592e-02, -2.84517370e-02,  1.08155357e+02],
                        [ 5.54330721e-02, -7.16394126e-01, -1.92132026e-01,  1.52908997e+02],
                        [-1.28090577e-02, -1.93749443e-01,  7.18249738e-01, -1.18026794e+02],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
            }


            nii_mask = nib.Nifti1Image(np_masks.transpose(1, 2, 0), affine=affine[subject])
            nii_ress = nib.Nifti1Image(np_result.transpose(1, 2, 0), affine=affine[subject])

            nib.save(nii_mask, os.path.join(path_to_save, "Consensus"+ subject+".nii.gz"))
            nib.save(nii_ress,  os.path.join(path_to_save, classifier+ "Classifier"+ subject+".nii.gz"))

            """
            from scipy.io import savemat

            #savemat("ConsensusROGU", {'matlab_array': np_masks})
            #savemat("FuzzyClassifierGULE", {'matlab_array': ress})
            """


 