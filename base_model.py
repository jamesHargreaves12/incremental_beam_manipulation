import os
import random
from math import log
from time import time
import numpy as np

from tensorflow.test import is_gpu_available
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import LSTM, TimeDistributed, Dense, Concatenate, Input, Embedding, CuDNNLSTM

from attention_keras.layers.attention import AttentionLayer
from utils import START_TOK


class TGEN_Model(object):
    def __init__(self, len_in, len_out, vsize_in, vsize_out, beam_size, cfg):
        self.batch_size = cfg["batch_size"]
        self.len_in = len_in
        self.len_out = len_out
        self.vsize_in = vsize_in
        self.vsize_out = vsize_out
        self.lstm_size = cfg["hidden_size"]
        self.embedding_size = cfg["embedding_size"]
        self.set_up_models(len_in, len_out, vsize_in, vsize_out, beam_size)

    def train(self, da_seq, text_seq, n_epochs, valid_da_seq, valid_text_seq, text_embedder, early_stop_point=5,
              minimum_stop_point=20):
        valid_onehot_seq = to_categorical(valid_text_seq, num_classes=self.vsize_out)
        text_onehot_seq = to_categorical(text_seq, num_classes=self.vsize_out)

        valid_losses = []
        rev_embed = text_embedder.embed_to_tok
        print('Valid Example:    {}'.format(" ".join([rev_embed[x] for x in valid_text_seq[0]]).replace('<>', '')))
        valid_pred = self.make_prediction(valid_da_seq[0], text_embedder, 1).replace(
            "<>", "")

        for ep in range(n_epochs):
            losses = 0
            start = time()
            batch_indexes = list(range(0, da_seq.shape[0] - self.batch_size, self.batch_size))
            random.shuffle(batch_indexes)
            for bi in batch_indexes:
                da_batch = da_seq[bi:bi + self.batch_size, :]
                text_batch = text_seq[bi:bi + self.batch_size, :]
                text_onehot_batch = text_onehot_seq[bi:bi + self.batch_size, :]
                self.full_model.train_on_batch([da_batch, text_batch[:, :-1]], text_onehot_batch[:, 1:, :])
                losses += self.full_model.evaluate([da_batch, text_batch[:, :-1]], text_onehot_batch[:, 1:, :],
                                                   batch_size=self.batch_size, verbose=0)
            if (ep + 1) % 1 == 0:
                valid_loss = 0
                for bi in range(0, valid_da_seq.shape[0] - self.batch_size, self.batch_size):
                    valid_da_batch = da_seq[bi:bi + self.batch_size, :]
                    valid_text_batch = text_seq[bi:bi + self.batch_size, :]
                    valid_onehot_batch = valid_onehot_seq[bi:bi + self.batch_size, :, :]
                    valid_loss += self.full_model.evaluate([valid_da_batch, valid_text_batch[:, :-1]],
                                                           valid_onehot_batch[:, 1:, :],
                                                           batch_size=self.batch_size, verbose=0)
                valid_losses.append(valid_loss)

                valid_pred = self.make_prediction_orig(valid_da_seq[0], text_embedder).replace(
                    "<>", "")
                # train_pred = self.make_prediction(da_seq[0], text_embedder).replace("<>", "")
                time_taken = time() - start
                train_loss = losses / da_seq.shape[0] * self.batch_size
                valid_loss = valid_loss / valid_da_seq.shape[0] * self.batch_size

                print("({:.2f}s) Epoch {} Loss: {:.4f} Valid: {:.4f} {}".format(time_taken, ep + 1,
                                                                                train_loss, valid_loss,
                                                                                valid_pred))
                if len(valid_losses) - np.argmin(valid_losses) > early_stop_point and len(
                        valid_losses) > minimum_stop_point:
                    return

    def make_prediction_orig(self, encoder_in, text_embedder):
        test_en = np.array([encoder_in])
        test_fr = np.array([text_embedder.tok_to_embed['<S>']])
        inf_enc_out = self.encoder_model.predict(test_en)
        enc_outs = inf_enc_out[0]
        enc_last_state = inf_enc_out[1:]
        dec_state = enc_last_state
        fr_in = test_fr
        fr_text = ''
        for i in range(20):
            out = self.decoder_model.predict([enc_outs, dec_state[0], dec_state[1], fr_in])
            dec_out, attention, dec_state = out[0], out[1], out[2:]
            dec_ind = np.argmax(dec_out, axis=-1)[0, 0]
            fr_in = np.array([dec_ind])
            fr_text += text_embedder.embed_to_tok[dec_ind] + ' '
        return fr_text


    def load_models_from_location(self, dir_name):
        self.full_model = load_model(os.path.join(dir_name, "full.h5"),
                                     custom_objects={'AttentionLayer': AttentionLayer})
        self.encoder_model = load_model(os.path.join(dir_name, "enc.h5"),
                                        custom_objects={'AttentionLayer': AttentionLayer})
        self.decoder_model = load_model(os.path.join(dir_name, "dec.h5"),
                                        custom_objects={'AttentionLayer': AttentionLayer})

    def save_model(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        self.full_model.save(os.path.join(dir_name, "full.h5"), save_format='h5')
        self.encoder_model.save(os.path.join(dir_name, "enc.h5"), save_format='h5')
        self.decoder_model.save(os.path.join(dir_name, "dec.h5"), save_format='h5')

    def beam_search_exapand(self, paths, end_tokens, enc_outs, beam_size):
        filled_paths = paths
        while len(filled_paths) < beam_size:
            filled_paths.append(paths[0])

        batch_enc_outs = np.array([enc_outs[0]] * beam_size)
        batch_dec_state_0 = []
        batch_dec_state_1 = []
        batch_tok = []
        for _, tok, dec_state in filled_paths:
            batch_tok.append([tok[-1]])
            batch_dec_state_0.append(dec_state[0][0])
            batch_dec_state_1.append(dec_state[1][0])
        inp = [batch_enc_outs, np.array(batch_dec_state_0), np.array(batch_dec_state_1), np.array(batch_tok)]

        out = self.decoder_model.predict(inp)
        dec_outs, dec_states = out[0], out[1:]

        new_paths = []
        tok_probs = []
        for p, dec_out, dec_state in zip(paths, dec_outs, dec_states):
            logprob, toks, dec_state = p
            if toks[-1] in end_tokens:
                new_paths.append((logprob, toks, dec_state))
                continue
            top_k = np.argsort(dec_out, axis=-1)[0][-beam_size:]
            tok_prob = dec_out[0][top_k]
            for new_tok, tok_prob in zip(top_k, tok_prob):
                tok_probs.append(tok_prob)
                new_paths.append((logprob + log(tok_prob), toks + [new_tok], dec_state))
        return new_paths, tok_probs

    def make_prediction(self, encoder_in, text_embedder, beam_size=1, prev_tok=None, max_length=20):
        if prev_tok is None:
            prev_tok = START_TOK
        test_en = np.array([encoder_in])
        test_fr = [text_embedder.tok_to_embed[prev_tok]]

        inf_enc_out = self.encoder_model.predict(test_en)
        enc_outs = inf_enc_out[0]
        enc_last_state = inf_enc_out[1:]
        paths = [(log(1.0), test_fr, enc_last_state)]

        end_embs = text_embedder.end_embs
        for i in range(max_length):
            new_paths,_ = self.beam_search_exapand(paths, end_embs, enc_outs, beam_size)

            paths = sorted(new_paths, key=lambda x: x[0], reverse=True)[:beam_size]
            if all([p[1][-1] in end_embs for p in paths]):
                break

        return " ".join(text_embedder.reverse_embedding(paths[0][1]))



    def set_up_models(self, len_in, len_out, vsize_in, vsize_out, inf_batch_size):
        lstm_type = CuDNNLSTM if is_gpu_available() else LSTM
        encoder_inputs = Input(batch_shape=(self.batch_size, len_in), name='encoder_inputs')
        decoder_inputs = Input(batch_shape=(self.batch_size, len_out - 1), name='decoder_inputs')

        embed_enc = Embedding(input_dim=vsize_in, output_dim=self.embedding_size)
        encoder_lstm = lstm_type(self.lstm_size, return_sequences=True, return_state=True, name='encoder_lstm')
        en_lstm_out = encoder_lstm(embed_enc(encoder_inputs))
        encoder_out = en_lstm_out[0]
        encoder_state = en_lstm_out[1:]

        embed_dec = Embedding(input_dim=vsize_out, output_dim=self.embedding_size)
        decoder_lstm = lstm_type(self.lstm_size, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_input_embeddings = embed_dec(decoder_inputs)
        # Attention layer
        attn_layer_Ws = AttentionLayer(name='attention_layer_t')

        # Ws
        attn_out_t, attn_states_t = attn_layer_Ws([encoder_out, decoder_input_embeddings])
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer_Ws')([decoder_input_embeddings, attn_out_t])
        dense_Ws = Dense(vsize_out, name='Ws')
        dense_time = TimeDistributed(dense_Ws, name='time_distributed_layer_Ws')
        decoder_lstm_in = dense_time(decoder_concat_input)

        de_lstm_out = decoder_lstm(decoder_lstm_in, initial_state=encoder_state)
        decoder_out = de_lstm_out[0]
        decoder_state = de_lstm_out[1:]

        decoder_concat_output = Concatenate(axis=-1, name='concat_layer_Wy')([decoder_out, attn_out_t])

        # Dense layer
        dense_Wy = Dense(vsize_out, name='Wy', activation='softmax')
        dense_time = TimeDistributed(dense_Wy, name='time_distributed_layer_Wy')
        decoder_pred = dense_time(decoder_concat_output)

        # Full model
        self.full_model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
        optimizer = Adam(lr=0.001)
        self.full_model.compile(optimizer=optimizer, loss='categorical_crossentropy')

        """ Encoder (Inference) model """
        encoder_inf_inputs = Input(batch_shape=(inf_batch_size, len_in), name='encoder_inf_inputs')
        en_lstm_out = encoder_lstm(embed_enc(encoder_inf_inputs))
        encoder_inf_out = en_lstm_out[0]
        encoder_inf_state = en_lstm_out[1:]

        self.encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_state])

        """ Decoder (Inference) model """
        dec_in = Input(batch_shape=(inf_batch_size, 1), name='decoder_word_inputs')
        encoder_out = Input(batch_shape=(inf_batch_size, len_in, self.lstm_size), name='encoder_inf_states')
        encoder_1 = Input(batch_shape=(inf_batch_size, self.lstm_size), name='decoder_init_1')
        encoder_2 = Input(batch_shape=(inf_batch_size, self.lstm_size), name='decoder_init_2')
        embed_dec_in = embed_dec(dec_in)

        # Ws
        attn_inf_out_t, attn_inf_states_t = attn_layer_Ws([encoder_out, embed_dec_in])
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([embed_dec_in, attn_inf_out_t])
        decoder_lstm_in = TimeDistributed(dense_Ws)(decoder_concat_input)

        de_lstm_out = decoder_lstm(decoder_lstm_in, initial_state=[encoder_1, encoder_2])
        decoder_inf_out = de_lstm_out[0]
        decoder_inf_state = de_lstm_out[1:]

        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_inf_out, attn_inf_out_t])
        decoder_inf_pred = TimeDistributed(dense_Wy)(decoder_inf_concat)
        self.decoder_model = Model(inputs=[encoder_out, encoder_1, encoder_2, dec_in],
                                   outputs=[decoder_inf_pred, decoder_inf_state])
