
# coding: utf-8

# In[1]:


import time
import numpy as np
import cv2
import os
import random
import subprocess as sp
from IPython.display import clear_output
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

os.chdir('../')
from utils import gen_and_show, show
from color import ColorDataset, ColorModel


# In[2]:


torch.backends.cudnn.benchmark = True
batch_size=24
seed = 87
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# In[3]:


# load data
data = ColorDataset()
training_data = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, drop_last=True, shuffle=True)

# load model
model = ColorModel()


# In[4]:


model.summary()


# In[ ]:


# Start training
epochs = 500
try:
    for epoch in range(epochs):
        #clear_output()
        gen_and_show(model, inputs=training_data, r=3)
        st = time.clock()
        logs = model.fit(training_data)
        et = time.clock()
        print ("[Epoch {:3d}] {}".format(epoch, ', '.join(map(str, logs))), et-st, 'sec')
        gen_and_show(model, inputs=training_data, r=3)
        model.save(epoch)
        #break
            
except KeyboardInterrupt:
    print ('KeyboardInterrupt')

