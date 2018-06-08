import torch
import numpy as np
import os
from utils import *
import cv2
from scipy.ndimage.filters import gaussian_filter


latent_dim = 128
colorizer = torch.load(os.path.join('saved_model', 'Colorizer-x.pt'))
#colorizer = torch.load(os.path.join('saved_model', 'Colorizer-290.pt'))
colorizer = colorizer.eval()
Illum = [0.4, 0.9, 0.6, 0.5, 0.5, 0.3, 0.5, 0.6, 0.3, 0.15, 0.4, 0.7]


def draw(tagtext, generator, std=0.2, input_imgs=None, resize=True, output_num=1):
    for hair in hair2idx:
        if hair in tagtext:
            break
        
    for eyes in eyes2idx:
        if eyes in tagtext:
            break
    
    #print(hair, eyes)
    hair = hair2idx[hair]
    eyes = eyes2idx[eyes]
    if eyes == 0 or eyes == 11:
        eyes = 1
    
    #print (type(input_imgs), type(np.array([])), type(input_imgs) == type(np.array([])))
    #output_num = 5 if type(input_imgs) != type(np.array([])) else len(input_imgs)
    
    if type(input_imgs) != type(np.array([])):
        # generate image
        noise = torch.randn(output_num, latent_dim) * std
        noise = createVariable(noise, True, True)
        I = torch.FloatTensor([Illum[hair] for _ in range(output_num)])
        I = createVariable(I, True, True)
        x = generator(noise, I)
    else:
        # has input image
        output_num = len(input_imgs)
        x = createVariable(torch.from_numpy(input_imgs).unsqueeze(1).float(), True, True)
    
    
    # paint color
    tag = torch.LongTensor([[hair, eyes] for _ in range(output_num)])
    hair = createVariable(tag[:, 0], True, True)
    eyes = createVariable(tag[:, 1], True, True)
    x = colorizer(hair, eyes, x).data.cpu().numpy().transpose(0, 2, 3, 1)
    
    imgs = np.array([toImage(i) for i in x])
    return imgs
    
    imgs = []
    for i, img in enumerate(x):
        img = np.clip(img, 0, 1)
        img = gaussian_filter(img, sigma=0.5)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC) if resize else img
        img = np.clip(img * 255 + 10, 0, 255).astype(np.uint8)
        imgs.append(img)
    return imgs