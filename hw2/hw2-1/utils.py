
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
import json
import sys
import os


# In[ ]:


import re
from collections import Counter
from keras.utils import to_categorical
from keras.callbacks import Callback
from config import *
from seq2seq_model import seq2seq_model


# In[ ]:


model, encoder_model, decoder_model, inf_model = seq2seq_model()

train_id_list = open('hw2_1_data/training_data/id.txt').read().split()
train_data = {i:np.load('hw2_1_data/training_data/feat/'+ i + '.npy') for i in train_id_list}
train_label = json.loads(open('hw2_1_data/training_label.json', 'r').read())
test_id_list = open('hw2_1_data/testing_data/id.txt').read().split()
test_data = {i:np.load('hw2_1_data/testing_data/feat/'+ i + '.npy') for i in test_id_list}
test_label = json.loads(open('hw2_1_data/testing_label.json', 'r').read())


# In[ ]:


def load_data():
    train_id_list = open('hw2_1_data/training_data/id.txt').read().split()
    train_data = {i:np.load('hw2_1_data/training_data/feat/'+ i + '.npy') for i in train_id_list}
    train_label = json.loads(open('hw2_1_data/training_label.json', 'r').read())
    test_id_list = open('hw2_1_data/testing_data/id.txt').read().split()
    test_data = {i:np.load('hw2_1_data/testing_data/feat/'+ i + '.npy') for i in test_id_list}
    test_label = json.loads(open('hw2_1_data/testing_label.json', 'r').read())


# In[ ]:


with open('vocab2idx', 'rb') as f:
    vocab2idx = pickle.load(f)
with open('idx2vocab', 'rb') as f:
    idx2vocab = pickle.load(f)
with open('correct_words', 'rb') as f:
    correct_words = pickle.load(f)


# In[ ]:


def words(text): return re.findall(r'\w+', text.lower())
WORDS = Counter(words(open('all_captions').read()))

def P(word):
    return - WORDS.get(word, 0)

def correction(word): 
    return max(candidates(word), key=P)

def candidates(word): 
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    return set(w for w in words if w in WORDS)

def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


# In[ ]:


def decode_sequence_reduce(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, vocab_size+1))
    target_seq[0, 0, vocab_size] = 1.
    
    stop_condition = False
    decoded_sentence = []
    last_word = ""
    last_last_word = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = idx2vocab[sampled_index]
        
        if sampled_word == last_word or sampled_word == last_last_word:
            output_tokens[0, -1, sampled_index] = 0
            sampled_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = idx2vocab[sampled_index]
        if sampled_word == last_word or sampled_word == last_last_word:
            output_tokens[0, -1, sampled_index] = 0
            sampled_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = idx2vocab[sampled_index]
        
        last_last_word = last_word
        last_word = sampled_word  
        decoded_sentence.append(sampled_word)
        if (len(decoded_sentence) >= out_length):# or sampled_word == "<pad>":
            stop_condition = True
        
        target_seq = np.zeros((1, 1, vocab_size+1))
        target_seq[0, 0, sampled_index] = 1.
        states_value = [h, c]

    return decoded_sentence


# In[ ]:


def make_node(state, prob, last, word, idx):
    seq = np.zeros((1, 1, vocab_size+1))
    seq[0, 0, idx] = 1.
    l = 0 if last == None else last['len']+1
    prob = 0 if last == None else prob+last['prob']
    node = {'state':state, 'seq':seq, 'prob':prob, 'last':last, 'len':l, 'word':word, 'idx':idx, 'next':[]}
    return node


# In[ ]:


def decode_sequence_beam(input_seq):
    states_value = encoder_model.predict(input_seq)
    init_node = make_node(states_value, 0, None, "<S>", vocab_size)
    queue = [init_node]
    leaf_nodes = []
    
    stop_condition = False
    decoded_sentence = []
    
    while len(queue) != 0:
        node = queue[0]
        if node['len'] >= out_length or node['word'] == '<pad>':
            leaf_nodes.append(node)
            queue = [] if len(queue) == 1 else queue[1:]
            break
        target_seq = node['seq']
        states_value = node['state']
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        for j in range(2):
            sampled_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = idx2vocab[sampled_index]
            if sampled_word != node['word']:
                new_node = make_node([h, c], output_tokens[0, -1, sampled_index], node, sampled_word, sampled_index)
                node['next'].append(new_node)
                queue.append(new_node)
            output_tokens[0, -1, sampled_index] = 0
        queue = queue[1:]
    
    # start search
    max_prob = 0
    for node in leaf_nodes:
        tmp = node['prob']/node['len']
        if tmp > max_prob:
            max_prob = tmp
            target_node = node
    
    while target_node['last'] != None:
        decoded_sentence.append(target_node['word'])
        target_node = target_node['last']
    
    return decoded_sentence[::-1]


# In[ ]:


def gen_output(data, id_list, decode, output_fn='output.txt'):
    with open(output_fn, 'w', encoding='utf-8') as f:
        for i in id_list:
            input_seq = np.array([data[i]])
            decoded_sentence = decode_sequence_reduce(input_seq)
            out = []
            last = ""
            for j in decoded_sentence:
                if j == "<S>" or j == last:
                    continue
                elif j == '<pad>':
                    break
                last = j
                out.append(j)
            
            out = i + ',' + " ".join(out) + '\n'
            f.write(out)


# In[ ]:


def cal_bleu(label):
    from hw2_1_data.bleu_eval import BLEU
    output = "output.txt"
    result = {}
    with open(output, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma+1:]
            result[test_id] = caption
    #count by the method described in the paper https://aclanthology.info/pdf/P/P02/P02-1040.pdf
    bleu=[]
    for item in label:
        score_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        
        score_per_video.append(BLEU(result[item['id']],captions,True))
        bleu.append(score_per_video[0])
    average = sum(bleu) / len(bleu)
    #print("Average bleu score is " + str(average))
    return average


# In[ ]:


def idx2data(idx, x, y, data, label):
    # x[idx] is id, y[idx][0] is label index, y[idx][1] is seq index
    encoder_input = data[x[idx]]
    decoder_input = label[y[idx][0]]['seq'][y[idx][1]]
    decoder_target = np.concatenate((decoder_input[1:], np.array([0], dtype='int32')))
    return encoder_input, decoder_input, decoder_target


# In[ ]:


def data_generator(data, targets):
    global train_data, train_label, batch_size, voacb_size
    idx = np.arange(len(data))
    while True:
        np.random.shuffle(idx)
        batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)] 
        
        for i in batches:
            encoder_inputs, decoder_inputs, decoder_targets = [], [], []
            for j in i:
                x, y, z = idx2data(j, data, targets, train_data, train_label)
                encoder_inputs.append(x)
                decoder_inputs.append(y)
                decoder_targets.append(z)
            
            encoder_inputs = np.array(encoder_inputs)
            decoder_inputs = to_categorical(decoder_inputs, num_classes=vocab_size+1).reshape(-1, out_length, vocab_size+1)
            decoder_targets = to_categorical(decoder_targets, num_classes=vocab_size+1).reshape(-1, out_length, vocab_size+1)
            yield ([encoder_inputs, decoder_inputs], decoder_targets)


# In[ ]:


def validation_generator(data, targets):
    global test_data, test_label, batch_size, voacb_size
    idx = np.arange(len(data))
    while True:
        np.random.shuffle(idx)
        batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)] 
        
        for i in batches:
            encoder_inputs, decoder_inputs, decoder_targets = [], [], []
            for j in i:
                x, y, z = idx2data(j, data, targets, test_data, test_label)
                encoder_inputs.append(x)
                decoder_inputs.append(y)
                decoder_targets.append(z)
            
            encoder_inputs = np.array(encoder_inputs)
            decoder_inputs = to_categorical(decoder_inputs, num_classes=vocab_size+1).reshape(-1, out_length, vocab_size+1)
            decoder_targets = to_categorical(decoder_targets, num_classes=vocab_size+1).reshape(-1, out_length, vocab_size+1)
            yield ([encoder_inputs, decoder_inputs], decoder_targets)


# In[ ]:


def idx2inf_data(idx, x, y, data, label):
    # x[idx] is id, y[idx][0] is label index, y[idx][1] is seq index
    encoder_input = data[x[idx]]
    decoder_input = label[y[idx][0]]['seq'][y[idx][1]]
    decoder_target = np.concatenate((decoder_input[1:], np.array([0], dtype='int32')))
    return encoder_input, decoder_input[0], decoder_target


# In[ ]:


def inf_data_generator(data, targets):
    global train_data, train_label, batch_size, voacb_size
    idx = np.arange(len(data))
    while True:
        np.random.shuffle(idx)
        batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)] 
        
        for i in batches:
            encoder_inputs, decoder_inputs, decoder_targets = [], [], []
            for j in i:
                x, y, z = idx2inf_data(j, data, targets, train_data, train_label)
                encoder_inputs.append(x)
                decoder_inputs.append(y)
                decoder_targets.append(z)
            
            encoder_inputs = np.array(encoder_inputs)
            decoder_inputs = to_categorical(decoder_inputs, num_classes=vocab_size+1).reshape(-1, 1, vocab_size+1)
            decoder_targets = to_categorical(decoder_targets, num_classes=vocab_size+1).reshape(-1, out_length, vocab_size+1)
            yield ([encoder_inputs, decoder_inputs], decoder_targets)


# In[ ]:


def inf_validation_generator(data, targets):
    global test_data, test_label, batch_size, voacb_size
    idx = np.arange(len(data))
    while True:
        np.random.shuffle(idx)
        batches = [idx[range(batch_size*i, min(len(data), batch_size*(i+1)))] for i in range(len(data)//batch_size+1)] 
        
        for i in batches:
            encoder_inputs, decoder_inputs, decoder_targets = [], [], []
            for j in i:
                x, y, z = idx2inf_data(j, data, targets, test_data, test_label)
                encoder_inputs.append(x)
                decoder_inputs.append(y)
                decoder_targets.append(z)
            
            encoder_inputs = np.array(encoder_inputs)
            decoder_inputs = to_categorical(decoder_inputs, num_classes=vocab_size+1).reshape(-1, 1, vocab_size+1)
            decoder_targets = to_categorical(decoder_targets, num_classes=vocab_size+1).reshape(-1, out_length, vocab_size+1)
            yield ([encoder_inputs, decoder_inputs], decoder_targets)


# In[ ]:


def preprocess(test_label, train_label):
    import string
    from keras.preprocessing.text import Tokenizer, text_to_word_sequence
    
    # count words
    t = Tokenizer()
    # fit_on_texts(texts)
    # texts: can be a list of strings, generator of strings, or a list of list of strings.
    for i in train_label:
        t.fit_on_texts(i['caption'])
    for i in test_label:
        t.fit_on_texts(i['caption'])

    # spelling correction
    for i in train_label:
        new = []
        for j in i['caption']:
            tmp = text_to_word_sequence(j)
            correct_list = []
            for k in range(len(tmp)):
                ignore_this_word = False
                for l in tmp[k]:
                    if l not in string.ascii_letters and l not in [" ", "'"]:
                        ignore_this_word = True
                        break
                if ignore_this_word:
                    continue
                #corrected = spell(tmp[k])
                corrected = correction(tmp[k])
                if corrected != tmp[k] and corrected in t.word_counts and t.word_counts[corrected] > t.word_counts[tmp[k]]*5 and t.word_counts[tmp[k]] < 10 and tmp[k][-2:] != "'s":
                    #print (tmp[k], t.word_counts[tmp[k]], corrected, t.word_counts[corrected], tmp)
                    correct_words[tmp[k]] = corrected
                    correct_list.append(corrected)
                else:
                    correct_list.append(tmp[k])

            new.append(" ".join(correct_list))
        i['caption'] = new
    
    t = Tokenizer()
    for i in train_label:
        t.fit_on_texts(i['caption'])
    for i in test_label:
        t.fit_on_texts(i['caption'])
    
    vocab_size = len(t.word_counts) + 1
    vocab2idx = dict((i, t.word_index[i]) for i in t.word_index)
    idx2vocab = dict((t.word_index[i], i) for i in t.word_index)
    idx2vocab[0] = "<pad>"
    idx2vocab[vocab_size] = "<S>"
    
    from keras.preprocessing.sequence import pad_sequences
    for i in train_label:
        seqs = t.texts_to_sequences(i['caption']) # input a list of strings
        seqs = [[vocab_size]+j for j in seqs] # put start symbol <S> at begining
        pad_seqs = pad_sequences(seqs, maxlen=out_length, dtype='int32', padding='post', truncating='post', value=0.0)
        i['seq'] = pad_seqs
    for i in test_label:
        seqs = t.texts_to_sequences(i['caption']) # input a list of strings
        seqs = [[vocab_size]+j for j in seqs] # put start symbol <S> at begining
        pad_seqs = pad_sequences(seqs, maxlen=out_length, dtype='int32', padding='post', truncating='post', value=0.0)
        i['seq'] = pad_seqs
    
    X = []
    Y = []
    for i, ii in enumerate(train_label):
        for j, jj in enumerate(ii['seq']):
            X.append(ii['id'])
            Y.append([i, j])

    X = np.array(X)
    Y = np.array(Y)
    
    X_test = []
    Y_test = []
    for i, ii in enumerate(test_label):
        for j, jj in enumerate(ii['seq']):
            X_test.append(ii['id'])
            Y_test.append([i, j])

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    return X, Y, X_test, Y_test


# In[ ]:


def plot_model():
    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)
    plot_model(encoder_model, to_file='encoder_model.png', show_shapes=True)
    plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True)


# In[ ]:


class MyCallback(Callback):
    def __init__(self):
        self.best_score = 0
        self.bleu_history = {'train':[], 'test':[]}
        self.saved_model = ""
        
    def on_epoch_end(self, epoch, logs={}):
        for i in self.params['metrics']:
            if i not in self.history:
                self.history[i] = []
            self.history[i].append(logs[i])
        
        gen_output(train_data, train_label, decode_sequence_reduce)
        show_outputs_and_score(train_label, 10)
        try:
            score = cal_bleu(train_label)
            score = round(score, 3)
            self.bleu_history['train'].append(score)
        except ZeroDivisionError:
            return
        print('\nTrain Bleu score: {}'.format(score))
        print ()
        
        gen_output(test_data, test_label, decode_sequence_reduce)
        show_outputs_and_score(test_label, 10)
        try:
            score = cal_bleu(test_label)
            score = round(score, 3)
            self.bleu_history['test'].append(score)
        except ZeroDivisionError:
            return
        print('Test Bleu score: {}\n'.format(score))
        
        if score > self.best_score:
            model.save_weights('model_{}.hdf5'.format(score))
            self.saved_model = 'model_{}.hdf5'.format(score)
            if self.best_score != 0:
                try:
                    os.remove('model_{}.hdf5'.format(self.best_score))
                except Exception as e:
                    print (str(e))
                    print ('model_{}.hdf5'.format(self.best_score), 'not found')
            self.best_score = score

