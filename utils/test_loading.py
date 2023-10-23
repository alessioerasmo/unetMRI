import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import os

class TestDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str):
        self.images_dir = images_dir
        self.masks_dir = mask_dir
        self.imgs = os.listdir(images_dir)
        self.imgs.sort()
        self.masks = os.listdir(mask_dir)
        self.masks.sort()
        assert len(self.imgs) == len(self.masks)
        self.length = len(self.masks)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        img_path = os.path.join(self.images_dir, self.imgs[index])
        mask_path = os.path.join(self.masks_dir, self.masks[index])
        
        return {
            'image': torch.as_tensor(np.array(Image.open(img_path))),
            'mask': torch.as_tensor(np.array(Image.open(mask_path)))
        }


    def getlist(self):
        return self.imgs
    

if __name__ == '__main__':
    test = TestDataset("data/test_imgs","data/test_masks" )
    import matplotlib.pyplot as plt

    for i in range(len(test)):
        image, mask = test[i]['image'], test[i]['mask']
        
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(image)
        axarr[1].imshow(mask)
        #axarr[2].imshow(np.array(image)-(np.array(mask)*255))
        plt.show()    
    