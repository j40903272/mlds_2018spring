import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from dataset import Conversation, DatasetIterator
from model_seq2seq import EncoderRNN, DecoderRNN


def train(x, y, l, encoder, decoder, enc_optim, dec_optim, criterion, teacher_forcing_ratio):

    x = [i for _,i in sorted(zip(l,x), reverse=True)]
    l.sort(reverse=True)
    
    x = Variable(torch.LongTensor(x)).cuda()
    y = Variable(torch.LongTensor(y)).cuda()

    enc_hid = encoder.initHidden(x.size(0))
    enc_out_hist = []
    loss = 0

    enc_out, enc_hid = encoder(x, l, enc_hid)

    dec_input = torch.LongTensor([[1]]*x.size(0)).cuda()
    dec_hid = (enc_hid[0, :, :] + enc_hid[1, :, :]).unsqueeze(0)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    for i in range(74):
        dec_out, dec_hid = decoder(dec_input, dec_hid, enc_out)
        loss += criterion(dec_out.squeeze(1), y[:, i])
        
        if use_teacher_forcing:
            dec_input = y[:, i:i+1]
        else:
            topv, topi = dec_out.topk(1)
            dec_input = topi.squeeze(1).detach()

    enc_optim.zero_grad()
    dec_optim.zero_grad()
    loss.backward()
    enc_optim.step()
    dec_optim.step()

    return loss.data[0]

def main():
    data = Conversation('./mlds_hw2_2_data/clr_conversation2.txt', 'vocab.txt')
    iterator = DatasetIterator(data)
    print('Done loading dataset')

    encoder = EncoderRNN(3000, 128).cuda()
    decoder = DecoderRNN(3000, 128, 3000).cuda()

    enc_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=1e-3)
    dec_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=1e-3)

    criterion = nn.NLLLoss()
    
    for i in range(750000):
        x, y, l = iterator.next_batch(100)
        l = train(x, y, l, encoder, decoder, enc_optim, dec_optim, criterion, 1)
        if i % 1000 == 0:
            print(f'iteration {i+1} loss = {l}')
    
        if i % 10000 == 9999:
            torch.save(encoder.state_dict(), open('encoderall.pt', 'wb'))
            torch.save(decoder.state_dict(), open('decoderall.pt', 'wb'))

if __name__ == '__main__':
    main()
