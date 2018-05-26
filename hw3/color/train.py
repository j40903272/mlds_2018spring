import sys
sys.path.append('..')

from utils import gen_and_show()
from dataset import ColorDataset
from model import ColorModel

import torch
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
import pickle
import itertools as it
import random
import os
from PIL import Image


torch.backends.cudnn.benchmark = True
batch_size=24
seed = 87
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

MODEL = os.getenv('MODEL') or ''

print("[*] Loading dataset")
data = ColorDataset(FLAGS.dataFolder)
print("[#] Loaded %d data" % len(data))
train = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, drop_last=True, shuffle=True)

print("[*] Building model")
model = ColorModel()
if torch.cuda.is_available():
    model = model.cuda()

print("[*] Start training")
bar = trange(500)
try:
    for epoch in bar:
        pred = model.predict([[0, 0]]*10, train)
        imgs, = zip(*pred)
        for i, x in enumerate(imgs):
            img, org = toImage(x)
            img.save(os.path.join(MODEL, 'output', 'cnorm', '%d-%d.jpg' % (epoch, i)))
            org.save(os.path.join(MODEL, 'output', 'corig', '%d-%d.jpg' % (epoch, i)))
        logs = model.fit(train)
        logs = ', '.join(map(str, logs))
        tqdm.write("[Epoch {:3d}] {}".format(epoch, logs))
        with open(os.path.join(FLAGS.dataFolder, 'pcom1', 'Colorizer-%d.pt' % epoch), 'wb') as f:
            torch.save(model.G, f)
        #  with open(os.path.join(FLAGS.dataFolder, 'training', 'Discriminator-%d.pt' % epoch), 'wb') as f:
            #  torch.save(model.D, f)
except KeyboardInterrupt:
    pass
bar.close()

with open(os.path.join(MODEL, 'output', 'Colorizer.pt'), 'wb') as f:
    torch.save(model.G, f)
