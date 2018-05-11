# -*- coding: utf-8 -*-
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from model_seq2seq import EncoderRNN, DecoderRNN

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


def load_vocab(path='vocab.txt'):
    wtoi = {}
    itow = []

    vocab = open(path, 'r', encoding='utf-8').read().splitlines()
    for i in vocab:
        idx, w = i.split()
        idx = int(idx)
        wtoi[w] = idx
        itow.append(w)

    return wtoi, itow

def load_test_data(wtoi, path='mlds_hw2_2_data/test_input.txt'):
    corpus = open(path, 'r', encoding='utf-8').read().splitlines()
    sentences = []
    lengths = []
    for s in corpus:
        seq = []
        for c in s:
            if c != ' ':
                if c in wtoi:
                    seq.append(wtoi[c])
                else:
                    seq.append(wtoi['<UNK>'])

        seq.append(wtoi['<EOS>'])
        lengths.append(len(seq))
        for j in range(74-len(seq)):
            seq.append(0)
        sentences.append(seq)
    sentences = np.array(sentences)
    return sentences, lengths

def test(x, l, encoder, decoder, itow):
    x = Variable(torch.LongTensor(x)).cuda()
    enc_hid = encoder.initHidden(1)

    enc_out, enc_hid = encoder(x, l, enc_hid)

    dec_input = torch.LongTensor([[1]]).cuda()
    dec_hid = (enc_hid[0, :, :] + enc_hid[1, :, :]).unsqueeze(0)

    y = []

    for i in range(74):
        dec_out, dec_hid = decoder(dec_input, dec_hid, enc_out)
        
        topv, topi = dec_out.topk(1)
        dec_input = topi.squeeze(1).detach()
        idx = dec_input.cpu().data.tolist()[0][0]
        if itow[idx] == '<EOS>':
            break
        if itow[idx] == '<UNK>':
            continue
        y.append(itow[idx])
    return ''.join(y)

def main():
    wtoi, itow = load_vocab()
    sentences, lengths = load_test_data(wtoi, sys.argv[1])
    
    encoder = EncoderRNN(3000, 512).cuda()
    decoder = DecoderRNN(3000, 512, 3000).cuda()

    encoder.load_state_dict(torch.load('encoder.pt'))
    decoder.load_state_dict(torch.load('decoder.pt'))
    print('done loading model.')
    f = open(sys.argv[2], 'w', encoding='utf8')

    for i in range(sentences.shape[0]):
        y = test(sentences[i:i+1], lengths[i:i+1], encoder, decoder, itow)
        f.write(y+'\n')

    f.close()


if __name__ == '__main__':
    main()
