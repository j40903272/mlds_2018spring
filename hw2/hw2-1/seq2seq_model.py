
# coding: utf-8

# In[ ]:


from config import *
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Permute, Reshape, merge


# In[ ]:


from keras.layers import Lambda
from keras import backend as K


# In[ ]:


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    TIME_STEPS = int(inputs.shape[1]) if inputs.shape[1] == in_length else out_length
    
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


# In[ ]:


def seq2seq_model():
    encoder_inputs = Input(shape=(in_length, input_dim), name='encoder_input')
    encoder_inputs_attn = attention_3d_block(encoder_inputs)
    encoder_inputs_drop = Dropout(drop_rate)(encoder_inputs_attn)#####
    encoder = LSTM(latent_dim, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs_drop)
    encoder_states = [state_h, state_c]
    decoder_inputs = Input(shape=(None, vocab_size+1), name='decoder_input')
    decoder_inputs_drop = Dropout(drop_rate)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs_drop, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size+1, activation='softmax', name='softmax_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    # model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    # encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # decoder
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )
    # inference
    states = encoder_states
    decoder_inputs = Input(shape=(1, vocab_size+1), name='inf_decoer_inputs')
    # reinjecting the decoder's predictions into the decoder's input, just like we were doing for inference.
    all_outputs = []
    inputs = decoder_inputs

    for _ in range(out_length):
        outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
        outputs = decoder_dense(outputs)
        all_outputs.append(outputs)
        inputs = outputs
        states = [state_h, state_c]

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
    inf_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    inf_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    
    return model, encoder_model, decoder_model, inf_model

