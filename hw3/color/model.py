import torch
from torch.autograd import Variable, grad
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
import os

from Net import Colorizer
from utils import createVariable, toList, count_parameters, show
from Components import History

class ColorModel(object):
    def __init__(self, *args, **kwargs):
        self.C = Colorizer()
        self.optimC = optim.Adam((p for p in self.C.parameters() if p.requires_grad), lr=6e-4, betas=(0.5, 0.999))
        self.use_cuda = False
        self.step = 0
        self.name = "color"
        self.Y_coff = [0.299, 0.587, 0.114]
        self.memory = []
        if torch.cuda.is_available():
            self.use_cuda = True
            self.cuda()

    def fit(self, dataset):
        
        Loss = History.Average('CL')
        for i, (x, y) in enumerate(dataset):
            self.step += 1
            print (self.step, Loss, end='\r')
            
            batch_size = y.size(0)
            x, y = [createVariable(z, self.use_cuda) for z in [x, y]]
            Y = sum(x[...,i] * self.Y_coff[i] for i in range(3))
            Y = Y.unsqueeze(1)
            tmp = x.permute(0, 3, 1, 2)

            # lr decay
            if self.step % 10000 == 0:
                for param_group in self.optimC.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

            self.optimC.zero_grad()
            self.C.train()
            closs = 0
            
            x = self.C(y[:, 0], y[:, 1], Y)
            #print ('mse', x.shape, tmp.shape)
            loss = F.mse_loss(x, tmp)
            closs += loss.data.cpu().numpy().tolist()[0]
            loss.backward()

            Loss.append(closs)
            torch.nn.utils.clip_grad_norm(self.C.parameters(), 1)
            self.optimC.step()
            
        return [Loss,]

    def predict(self, tags, data, show_org=False):
        self.C.train()
        output, imgs = [], []
        l = 0

        while l < len(tags):
            for x, y in data:
                imgs.append(x)
                l += x.size(0)
                if l >= len(tags):
                    break
                    
        
        c = torch.cat(imgs, 0) if len(imgs) > 1 else imgs[0]
        Y = sum(c[...,i] * self.Y_coff[i] for i in range(3))
        Y = Y.unsqueeze(1)
        I = Y

        for tag, Y in zip(tags, Y):
            tag = torch.LongTensor(tag).unsqueeze(0)
            Y = Y.unsqueeze(0)

            # Training Generator
            Y = createVariable(Y, self.use_cuda, True)
            hair = createVariable(tag[:, 0], self.use_cuda, True)
            eyes = createVariable(tag[:, 1], self.use_cuda, True)
            #print (hair.shape, eyes.shape, Y.shape)
            x = self.C(hair, eyes, Y)
            output.append(x.data.cpu().numpy()[0])

        return np.array(output).transpose(0, 2, 3, 1), c.numpy()

    def cuda(self):
        self.use_cuda = True
        self.C = self.C.cuda()
        return self

    def cpu(self):
        self.use_cuda = False
        self.C = self.C.cpu()
        return self
    
    def summary(self):
        print ()
        print ('colorizer', count_parameters(self.C), 'params')
        print (self.C)
        print ()
        '''
        for name, p in model.G.named_parameters():
            if p.requires_grad:
                print (name, p.numel())
        for name, p in model.D.named_parameters():
            if p.requires_grad:    
                print (name, p.numel())
        '''
    
    def load(self, path):
        self.C = torch.load(path)
        
    
    def save(self, epoch, path='saved_model'):
        with open(os.path.join(path, 'Colorizer-%d.pt' % epoch), 'wb') as f:
            torch.save(self.C, f)