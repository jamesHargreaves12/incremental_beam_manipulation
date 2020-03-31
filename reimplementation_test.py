import random
import sys
import os
from time import time

import numpy as np
from keras.utils import to_categorical

from tensorflow.test import is_gpu_available
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import LSTM, TimeDistributed, Dense, Concatenate, Input, Embedding, CuDNNLSTM

from attention_keras.layers.attention import AttentionLayer

sys.path.append(os.path.join(os.getcwd(), 'tgen'))

from tgen.futil import read_trees_or_tokens
from tgen.embeddings import DAEmbeddingSeq2SeqExtract, TokenEmbeddingSeq2SeqExtract
from tgen.futil import read_das


def get_model(batch_size, in_max_len, out_max_len, in_vsize, out_vsize, hidden_size, embedding_size):
    lstm_type = CuDNNLSTM if is_gpu_available() else LSTM
    decoder_in_size = out_vsize
    encoder_inputs = Input(batch_shape=(batch_size, in_max_len), name='encoder_inputs')
    decoder_inputs = Input(batch_shape=(batch_size, out_max_len - 1), name='decoder_inputs')

    embed_enc = Embedding(input_dim=in_vsize, output_dim=embedding_size)
    encoder_lstm = lstm_type(hidden_size, return_sequences=True, return_state=True, name='encoder_lstm')
    en_lstm_out = encoder_lstm(embed_enc(encoder_inputs))
    encoder_out = en_lstm_out[0]
    encoder_state = en_lstm_out[1:]

    embed_dec = Embedding(input_dim=out_vsize, output_dim=embedding_size)
    decoder_lstm = lstm_type(hidden_size, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_input_embeddings = embed_dec(decoder_inputs)
    # Attention layer
    attn_layer_Ws = AttentionLayer(name='attention_layer_t')

    # Ws
    attn_out_t, attn_states_t = attn_layer_Ws([encoder_out, decoder_input_embeddings])
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer_Ws')([decoder_input_embeddings, attn_out_t])
    dense_Ws = Dense(decoder_in_size, name='Ws')
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
    dense_Wy = Dense(out_vsize, name='Wy',  activation='softmax')
    dense_time = TimeDistributed(dense_Wy, name='time_distributed_layer_Wy')
    decoder_pred = dense_time(decoder_concat_output)

    # Full model
    full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    optimizer = Adam(lr=0.001)
    full_model.compile(optimizer=optimizer, loss='categorical_crossentropy')

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


def train(full_model, da_seq, text_seq, batch_size, n_epochs, text_vocab_size, valid_da_seq, valid_text_seq, early_stop_point=5):
    """ Training the model """
    valid_onehot_seq = to_categorical(valid_text_seq, num_classes=text_vocab_size)
    text_onehot_seq = to_categorical(text_seq, num_classes=text_vocab_size)

    valid_losses = []
    for ep in range(n_epochs):
        losses = 0
        start = time()
        batch_indexes = list(range(0, da_seq.shape[0] - batch_size, batch_size))
        random.shuffle(batch_indexes)
        for bi in batch_indexes:
            da_batch = da_seq[bi:bi + batch_size, :]
            text_batch = text_seq[bi:bi + batch_size, :]
            text_onehot_batch = text_onehot_seq[bi:bi + batch_size, :]
            full_model.train_on_batch([da_batch, text_batch[:, :-1]], text_onehot_batch[:, 1:, :])
            losses += full_model.evaluate([da_batch, text_batch[:, :-1]], text_onehot_batch[:, 1:, :],
                                          batch_size=batch_size, verbose=0)
        if (ep + 1) % 1 == 0:
            valid_loss = 0
            for bi in range(0, valid_da_seq.shape[0] - batch_size, batch_size):
                valid_da_batch = da_seq[bi:bi + batch_size, :]
                valid_text_batch = text_seq[bi:bi + batch_size, :]
                valid_onehot_batch = valid_onehot_seq[bi:bi + batch_size, :, :]
                valid_loss += full_model.evaluate([valid_da_batch, valid_text_batch[:, :-1]],
                                                  valid_onehot_batch[:, 1:, :],
                                                  batch_size=batch_size, verbose=0)
            valid_losses.append(valid_loss)
            print("({:.2f}s) Epoch {} Loss: {:.4f} Valid: {:.4f}".format(time() - start, ep + 1,
                                                                         losses / da_seq.shape[0] * batch_size,
                                                                         valid_loss / valid_da_seq.shape[
                                                                             0] * batch_size))
            if len(valid_losses) - np.argmin(valid_losses) > early_stop_point:
                return

use_size = 100
valid_size = 100
epoch = 100
batch_size = 20
hidden_size = 128

das = read_das("tgen/e2e-challenge/input/train-das.txt")
trees = [[('<Start>', None)] + x + [("<End>", None)] for x in
         read_trees_or_tokens("tgen/e2e-challenge/input/train-text.txt", 'tokens', 'en', '')]
print(das[0], trees[0])
das = das[:use_size + valid_size]
trees = trees[:use_size + valid_size]
da_embs = DAEmbeddingSeq2SeqExtract(cfg={'sort_da_emb': True})
tree_embs = TokenEmbeddingSeq2SeqExtract(cfg={'max_sent_len': 80})

da_vsize = da_embs.init_dict(das)
text_vsize = tree_embs.init_dict(trees)
da_max_len = da_embs.get_embeddings_shape()[0]
text_max_len = tree_embs.get_embeddings_shape()[0]
print(da_vsize, text_vsize, da_max_len, text_max_len)


train_enc = np.array([da_embs.get_embeddings(da) for da in das])
train_dec = np.array([tree_embs.get_embeddings(tree) for tree in trees])

full_model, infer_enc_model, infer_dec_model = get_model(batch_size, da_max_len, text_max_len, da_vsize, text_vsize,
                                                         hidden_size, 50)

full_model_save_path = 'models/reimp_save_test__.tf'
if os.path.exists(full_model_save_path + '.index'):
    print("Loading model from file")
    full_model.load_weights(full_model_save_path)
else:
    print("Training model")
    train(full_model, train_enc[:-valid_size], train_dec[:-valid_size], batch_size, epoch, text_vsize,
          train_enc[-valid_size:], train_dec[-valid_size:])
    # full_model.save_weights(full_model_save_path, save_format='tf')



""" Inferring with trained model """
print("Testing greedy inference")
test_en = np.array([train_enc[-valid_size]])
test_fr = np.array([tree_embs.GO])

inf_enc_out = infer_enc_model.predict(test_en)
enc_outs = inf_enc_out[0]
enc_last_state = inf_enc_out[1:]

dec_state = enc_last_state
attention_weights = []
fr_text = ''

fr_in = test_fr
for i in range(20):
    out = infer_dec_model.predict([enc_outs, dec_state[0], dec_state[1], fr_in])
    dec_out, attention, dec_state = out[0], out[1], out[2:]
    dec_ind = np.argmax(dec_out, axis=-1)[0, 0]

    fr_in = np.array([dec_ind])

    attention_weights.append((dec_ind, attention))
    fr_text += tree_embs.rev_dict[dec_ind] + ' '
print('Aim:    {}'.format(" ".join([x[0] for x in trees[-valid_size]])))
print('Output: {}'.format(fr_text))

""" Attention plotting """
# plot_attention_weights(test_en_seq, attn_weights, en_index2word, fr_index2word, base_dir=base_dir)
