import sys
import os
from time import time

import numpy as np
from keras.utils import to_categorical
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import LSTM, TimeDistributed, Dense, Concatenate, Input, Embedding

from attention_keras.layers.attention import AttentionLayer

sys.path.append(os.path.join(os.getcwd(), 'tgen'))

from tgen.futil import read_trees_or_tokens
from tgen.embeddings import DAEmbeddingSeq2SeqExtract, TokenEmbeddingSeq2SeqExtract
from tgen.futil import read_das


def get_model(batch_size, in_max_len, out_max_len, in_vsize, out_vsize, hidden_size, embedding_size):
    decoder_in_size = out_vsize
    encoder_inputs = Input(batch_shape=(batch_size, in_max_len), name='encoder_inputs')
    decoder_inputs = Input(batch_shape=(batch_size, out_max_len - 1), name='decoder_inputs')

    embed_enc = Embedding(input_dim=in_vsize, output_dim=embedding_size)
    encoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True, name='encoder_lstm')
    en_lstm_out = encoder_lstm(embed_enc(encoder_inputs))
    encoder_out = en_lstm_out[0]
    encoder_state = en_lstm_out[1:]

    embed_dec = Embedding(input_dim=out_vsize, output_dim=embedding_size)
    decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_input_embeddings = embed_dec(decoder_inputs)
    # Attention layer
    attn_layer_Ws = AttentionLayer(name='attention_layer_t')

    # Ws
    attn_out_t, attn_states_t = attn_layer_Ws([encoder_out, decoder_input_embeddings])
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer_Ws')([decoder_input_embeddings, attn_out_t])
    dense_Ws = Dense(decoder_in_size, activation='softmax', name='Ws')
    dense_time = TimeDistributed(dense_Ws, name='time_distributed_layer_Ws')
    decoder_lstm_in = dense_time(decoder_concat_input)

    de_lstm_out = decoder_lstm(decoder_lstm_in, initial_state=encoder_state)
    decoder_out = de_lstm_out[0]
    decoder_state = de_lstm_out[1:]

    # Attention layer Wy
    attn_layer_Wy = AttentionLayer(name='attention_layer_t1')
    attn_out_t1, attn_states_t1 = attn_layer_Wy([encoder_out, decoder_out])
    decoder_concat_output = Concatenate(axis=-1, name='concat_layer_Wy')([decoder_out, attn_out_t1])

    # Dense layer
    dense_Wy = Dense(out_vsize, name='Wy')
    dense_time = TimeDistributed(dense_Wy, name='time_distributed_layer_Wy')
    decoder_pred = dense_time(decoder_concat_output)

    # Full model
    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    full_model.compile(optimizer='adam', loss='categorical_crossentropy')

    full_model.summary()

    """ Inference model """
    batch_size = 1

    """ Encoder (Inference) model """
    encoder_inf_inputs = Input(batch_shape=(batch_size, in_max_len), name='encoder_inf_inputs')
    en_lstm_out = encoder_lstm(embed_enc(encoder_inf_inputs))
    encoder_inf_out = en_lstm_out[0]
    encoder_inf_state = en_lstm_out[1:]

    encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_state])

    """ Decoder (Inference) model """
    dec_in = Input(batch_shape=(batch_size, 1), name='decoder_word_inputs')
    encoder_out = Input(batch_shape=(batch_size, in_max_len, hidden_size), name='encoder_inf_states')
    encoder_1 = Input(batch_shape=(batch_size, hidden_size), name='decoder_init_1')
    encoder_2 = Input(batch_shape=(batch_size, hidden_size), name='decoder_init_2')
    embed_dec_in = embed_dec(dec_in)

    # Ws
    attn_inf_out_t, attn_inf_states_t = attn_layer_Ws([encoder_out, embed_dec_in])
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([embed_dec_in, attn_inf_out_t])
    decoder_lstm_in = TimeDistributed(dense_Ws)(decoder_concat_input)

    de_lstm_out = decoder_lstm(decoder_lstm_in, initial_state=[encoder_1, encoder_2])
    decoder_inf_out = de_lstm_out[0]
    decoder_inf_state = de_lstm_out[1:]

    attn_inf_out_t1, attn_inf_states_t1 = attn_layer_Wy([encoder_out, decoder_inf_out])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out_t1])
    decoder_inf_pred = TimeDistributed(dense_Wy)(decoder_inf_concat)
    decoder_model = Model(inputs=[encoder_out, encoder_1, encoder_2, dec_in],
                          outputs=[decoder_inf_pred, attn_inf_states_t1, decoder_inf_state])

    encoder_model.summary()
    decoder_model.summary()
    return full_model, encoder_model, decoder_model


def train(full_model, en_seq, fr_seq, batch_size, n_epochs, fr_vsize):
    """ Training the model """

    for ep in range(n_epochs):
        losses = []
        start = time()
        for bi in range(0, en_seq.shape[0] - batch_size, batch_size):
            en_batch = en_seq[bi:bi + batch_size, :]
            fr_batch = fr_seq[bi:bi + batch_size, :]
            fr_onehot_seq = to_categorical(fr_batch, num_classes=fr_vsize)

            full_model.train_on_batch([en_batch, fr_batch[:, :-1]], fr_onehot_seq[:, 1:, :])

            l = full_model.evaluate([en_batch, fr_batch[:, :-1]], fr_onehot_seq[:, 1:, :],
                                    batch_size=batch_size, verbose=0)

            losses.append(l)
        if (ep + 1) % 1 == 0:
            print("Time: {} Loss in epoch {}: {}".format(time() - start, ep + 1, np.mean(losses)))


use_size = 42000
epoch = 20
batch_size = 20
hidden_size = 128

das = read_das("tgen/e2e-challenge/input/train-das.txt")
trees = [[('<Start>', None)] + x + [("<End>", None)] for x in
         read_trees_or_tokens("tgen/e2e-challenge/input/train-text.txt", 'tokens', 'en', '')]
print(das[0], trees[0])
das = das[:use_size]
trees = trees[:use_size]
da_embs = DAEmbeddingSeq2SeqExtract(cfg={'sort_da_emb': True})
tree_embs = TokenEmbeddingSeq2SeqExtract(cfg={'max_sent_len': 80})

en_vsize = da_embs.init_dict(das)
fr_vsize = tree_embs.init_dict(trees)
en_timesteps = da_embs.get_embeddings_shape()[0]
fr_timesteps = tree_embs.get_embeddings_shape()[0]
print(en_vsize, fr_vsize, en_timesteps, fr_timesteps)

# prepare training batches
train_enc = np.array([da_embs.get_embeddings(da) for da in das])
train_dec = np.array([tree_embs.get_embeddings(tree) for tree in trees])

full_model, infer_enc_model, infer_dec_model = get_model(batch_size, en_timesteps, fr_timesteps, en_vsize, fr_vsize,
                                                         hidden_size, 50)

train(full_model, train_enc, train_dec, batch_size, 5, fr_vsize)

""" Inferring with trained model """
test_en = train_enc[:1]
test_fr = np.array([tree_embs.GO])
print(tree_embs.GO)
print(test_fr)
print('Translating: {}'.format(test_en))
# test_en_onehot_seq = to_categorical(test_en, num_classes=en_vsize)

# print(test_en_onehot_seq.shape)
inf_enc_out = infer_enc_model.predict(test_en)
enc_outs = inf_enc_out[0]
enc_last_state = inf_enc_out[1:]
print("enc_out", enc_outs.shape)
print("enc_1", enc_last_state[0].shape)
print("enc_2", enc_last_state[1].shape)
# print("dec_in", test_fr_onehot_seq.shape)
dec_state = enc_last_state
attention_weights = []
fr_text = ''

fr_in = test_fr
for i in range(20):
    out = infer_dec_model.predict([enc_outs, dec_state[0], dec_state[1], fr_in])
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
    dec_out, attention, dec_state = out[0], out[1], out[2:]
    dec_ind = np.argmax(dec_out, axis=-1)[0, 0]
    print("Value", tree_embs.rev_dict[dec_ind])
    fr_in = np.array([dec_ind])

    attention_weights.append((dec_ind, attention))
    fr_text += tree_embs.rev_dict[dec_ind] + ' '
print('\tOutput: {}'.format(fr_text))

""" Attention plotting """
# plot_attention_weights(test_en_seq, attn_weights, en_index2word, fr_index2word, base_dir=base_dir)
