from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import torchvision.transforms.functional as F

def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    
    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype = np.float32)
    return gaussian_noise_imgs


img = Image.open("data/test_imgs/MSSEG2016_01042GULE_348.jpg")
tensor_img = torch.tensor(np.array(img))
contr = F.adjust_brightness(img, 0.5)

print(np.average(np.array(img)), "max value: ", np.array(img).max())
print(np.average(np.array(contr)), "max value: ",np.array(contr).max() )



import random

while(True):

    brightness_factor = torch.empty(1).uniform_(0.5, 2).item()    
    contr = F.adjust_brightness(img, brightness_factor)
    contr = F.gaussian_blur(img, [3,3], random.random())

    noise = torch.randn(tensor_img.size()) * 0.5 + 0
    contr = (tensor_img + noise).numpy()

    # Genera due immagini di esempio (sostituisci queste con le tue immagini)
    image1 = img  # Immagine 1 casuale
    image2 = contr  # Immagine 2 casuale

    # Imposta le dimensioni della figura (larghezza x altezza)
    fig = plt.figure(figsize=(20, 12))  # Imposta le dimensioni desiderate
    
    # Crea una figura con due subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Mostra la prima immagine nel primo subplot
    ax1.imshow(image1, cmap='gray')
    ax1.set_title('original, max=' + str(np.array(image1).max()))

    # Mostra la seconda immagine nel secondo subplot
    ax2.imshow(image2, cmap='gray')
    ax2.set_title('brightenss, max='  + str(np.array(image2).max()))

    # Mostra la figura con entrambe le immagini
    plt.show()
