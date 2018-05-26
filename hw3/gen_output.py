import torch
import numpy as np
import sys, os
import random
from draw import draw
import matplotlib
matplotlib.use('Agg')

torch.backends.cudnn.benchmark = True
batch_size=64
seed = 878787
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    
generator = torch.load(os.path.join('saved_model', 'Generator-80.pt'))

def save_imgs(gen_imgs):
    import matplotlib.pyplot as plt
    r, c = 5, 5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i,j].axis('off')
            cnt += 1
    #fig.savefig("samples/cgan_original.png")
    fig.savefig("samples/cgan-%d.png" % seed)
    plt.close()

with open(sys.argv[1]) as f:
    gen_imgs = []
    for line in f:
        line = line.strip()
        idx, tagtext = line.split(',', 1)
        gen_imgs.append(draw(tagtext, generator, resize=True, std=0.3)[0])
        
    if len(gen_imgs) != 25:
        print ('[ERROR] generate', len(gen_imgs), 'images !')
    
    save_imgs(gen_imgs)
