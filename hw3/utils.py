import pickle
with open('./hair2idx', 'rb') as f:
    hair2idx = pickle.load(f)
with open('./eyes2idx', 'rb') as f:
    eyes2idx = pickle.load(f)


import numpy as np

import torch
from torch.autograd import Variable
from torch import Tensor

def createVariable(tensor, use_cuda, volatile=False, **kwargs):
    var = Variable(tensor, volatile=volatile, **kwargs)
    return var.cuda() if use_cuda else var

    
def toList(x):
    if isinstance(x, Variable):
        return x.data.cpu().numpy().tolist()
    if isinstance(x, Tensor):
        return x.cpu().numpy().tolist()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gen_and_show(model, r=5, c=5, inputs=None):
    if model.name == 'GAN':
        pred = model.predict([0.5]*(r*c))
        pred = sorted(pred, key=lambda p: p[1])
        imgs, _, _ = zip(*pred)
        show(imgs, r, c)
    elif model.name == 'color':
        pred = model.predict([[0, 0]]*(r*c), inputs, show_org=True)
        imgs, orgs = pred
        #print (imgs.shape, orgs.shape)
        #print (imgs[0], orgs[0])
        show(orgs, r, c, color=True)
        show(imgs, r, c, color=True)
    
    

def show(imgs, r, c, color=False):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(r, c, figsize=(10,10))
    cnt = 0
    for i in range(r):
        for j in range(c):
            img = toImage(imgs[cnt])
            if color:
                axs[i,j].imshow(img, cmap='gray')
            else:
                axs[i,j].imshow(img)
            axs[i,j].axis('off')
            cnt += 1
            
    plt.tight_layout()
    plt.show()

    
def illum(x):
    r, g, b = x[:,:,0], x[:,:,1], x[:,:,2]
    return (r * 0.299 + g * 0.587 + b * 0.114)

        
def toImage(x, resize=True):
    import numpy as np
    from PIL import Image
    from skimage import transform, filters
    
    if x.shape[2] != 3 and x.shape[2] != 1:
        x = x.transpose(1, 2, 0)
    if x.shape[2] != 3:
        #print (x.shape)
        x = x.reshape(x.shape[:2])
    #org = np.clip(x, 0, 1)
    #org = (org * 255).astype(np.uint8)
    #org = Image.fromarray(org)
    img = np.clip(x, 0, 1)
    img = filters.gaussian(img, sigma=0.5, multichannel=True)
    img = transform.resize(img, (64, 64), mode='reflect') if resize else img
    img = np.clip(img * 255 + 10, 0, 255).astype(np.uint8)
    return img