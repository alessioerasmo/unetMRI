import torch
from unet import UNet

# Model class must be defined somewhere
model = torch.load("checkpoints/checkpoint_epoch5.pth")


from PIL import Image
import numpy as np

image = np.array(Image.open("data/imgs/fff9b3a5373f_16.jpg"))

import matplotlib.pyplot as plt

print(image.shape)

model.forward(image)

