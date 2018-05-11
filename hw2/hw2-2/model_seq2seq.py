import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.emb = nn.Embedding(3000, 3000)
        self.emb.load_state_dict({'weight': torch.FloatTensor(np.identity(3000))})
        self.emb.weight.requires_grad = False
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
                input_size,
                hidden_size,
                n_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
                )

    def forward(self, input, lengths, hidden):
        input = self.emb(input)
        packed = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        output, hidden = self.gru(packed, hidden) # output: (bs, 15, 2h)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, hidden

    def initHidden(self, batch_size):
        return Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()

'''
class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_len = context.size(0)
        # (bs, out_len, hid) * (bs, in_len, hid) -> (bs, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)

        # (bs, out_len, in_len) * (bs, in_len, hid) -> (bs, out_len, hid)
        mix = torch.bmm(attn, context)
        
        # concat -> (bs, out_len, hid*2)
        combined = torch.cat((mix, output), dim=2)

        # out -> (bs, out_len, hid)
        # output = F.tanh(self.out(combined))

        return output, attn
'''      
        
class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.emb = nn.Embedding(3000, 3000)
        self.emb.load_state_dict({'weight': torch.FloatTensor(np.identity(3000))})
        self.emb.weight.requires_grad = False

        # self.attn = Attention(self.hidden_size)
        self.linear = nn.Linear(input_size, hidden_size)
        # self.attn = nn.Linear(hidden_size*2, 15)
        # self.attn_comb = nn.Linear(hidden_size*2, hidden_size)

        # self.dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(
                self.hidden_size,
                self.hidden_size,
                batch_first=True,
                bidirectional=False,
                dropout=dropout
                )
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, enc_out_hist=None):
        input = self.emb(input)
        '''
        output, hidden = self.lstm(input, hidden)
        output, attn = self.attn(output, enc_out_hist)
        output = self.out(output)
        output = F.softmax(output, dim=-1)
        '''
        input = self.linear(input) # (bs, 1, h)
        # concat = torch.cat((input.squeeze(1), hidden[0]), -1).unsqueeze(1) # (bs, 1, 2h)
        # attn_w = F.softmax(self.attn(concat), dim=-1) # (bs, 1, 15)
        # (bs, 1, 15) * (bs, 15, h) -> (bs, 1, h)
        # attn_applied = torch.bmm(attn_w, enc_out_hist)
        # output = torch.cat((input, attn_applied), -1) # (bs, 1, 2h)
        # output = self.attn_comb(output) # (bs, 1, h)
        # output = F.relu(output)
        output, hidden = self.gru(input, hidden)
        output = F.log_softmax(self.out(output), dim=-1)
        return output, hidden

    def initBOS(self, batch_size):
        return Variable(torch.LongTensor([[1]]*batch_size))
