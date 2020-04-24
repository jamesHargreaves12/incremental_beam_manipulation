import math
import os
import pickle
import random
from math import log
from time import time

import msgpack
import numpy as np
import yaml

from tensorflow.test import is_gpu_available
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import LSTM, TimeDistributed, Dense, Concatenate, Input, Embedding, CuDNNLSTM
from tqdm import tqdm

from attention_keras.layers.attention import AttentionLayer
from utils import START_TOK, get_hamming_distance, PAD_TOK, load_model_from_gpu


class TrainableReranker(object):
    def __init__(self, da_embedder, text_embedder, cfg_path):
        cfg = yaml.load(open(cfg_path, "r+"))
        self.embedding_size = cfg['embedding_size']
        self.lstm_size = cfg['hidden_size']
        self.batch_size = cfg['training_batch_size']
        self.text_embedder = text_embedder
        self.da_embedder = da_embedder
        self.save_location = cfg["reranker_loc"]
        self.model = None
        self.min_log_prob = cfg["default_min_log_prob"]
        self.max_log_prob = cfg["default_max_log_prob"]
        self.set_up_models()

    def set_up_models(self):
        len_text = self.text_embedder.length
        len_vtext = self.text_embedder.vocab_length
        len_da = self.da_embedder.length
        len_vda = self.da_embedder.vocab_length

        lstm_type = CuDNNLSTM if is_gpu_available() else LSTM

        text_inputs = Input(shape=(len_text,), name='text_inputs')
        da_inputs = Input(shape=(len_da,), name='da_inputs')
        log_probs_inputs = Input(shape=(1, ), name='log_probs_inputs')

        embed_text = Embedding(input_dim=len_vtext, output_dim=self.embedding_size)
        embed_da = Embedding(input_dim=len_vda, output_dim=self.embedding_size)

        text_lstm = lstm_type(self.lstm_size, return_sequences=True, return_state=True, name='text_lstm')
        da_lstm = lstm_type(self.lstm_size, return_sequences=True, return_state=True, name='da_lstm')

        text_lstm_out = text_lstm(embed_text(text_inputs))
        da_lstm_out = da_lstm(embed_da(da_inputs))

        h_n_text = text_lstm_out[1:]
        h_n_da = da_lstm_out[1:]
        in_logistic_layer = Concatenate(axis=-1, name='concat_layer_Wy')(h_n_text + h_n_da + [log_probs_inputs])

        hidden_logistic = Dense(128, activation='relu')(in_logistic_layer)
        output = Dense(1)(hidden_logistic)
        self.model = Model(inputs=[text_inputs, da_inputs, log_probs_inputs], outputs=output)
        optimizer = Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.model.summary()

    def get_valid_loss(self, valid_das_seqs, valid_text_seqs, valid_log_probs, valid_bleu_scores):
        valid_loss = 0
        for bi in range(0, valid_das_seqs.shape[0] - self.batch_size + 1, self.batch_size):
            valid_da_batch = valid_das_seqs[bi:bi + self.batch_size, :]
            valid_text_batch = valid_text_seqs[bi:bi + self.batch_size, :]
            lp_batch = valid_log_probs[bi:bi + self.batch_size, :]
            bleu_scores_batch = valid_bleu_scores[bi:bi + self.batch_size, :]
            valid_loss += self.model.evaluate([valid_text_batch, valid_da_batch,lp_batch], bleu_scores_batch,
                                              batch_size=self.batch_size, verbose=0)
        return valid_loss

    def train(self, text_seqs, das_seqs, bleu_scores, log_probs,  epoch, valid_size):
        min_valid_loss = math.inf
        epoch_since_minimum = 0
        valid_text_seqs = text_seqs[-valid_size:]
        valid_das = das_seqs[-valid_size:]
        valid_bleu_scores = bleu_scores[-valid_size:]
        valid_log_probs = log_probs[-valid_size:]
        text_seqs = text_seqs[:-valid_size]
        das_seqs = das_seqs[:-valid_size]
        bleu_scores = bleu_scores[:-valid_size]
        log_probs = log_probs[:-valid_size]

        for ep in range(epoch):
            start = time()
            losses = 0
            batch_indexes = list(range(0, text_seqs.shape[0] - self.batch_size, self.batch_size))
            random.shuffle(batch_indexes)
            for bi in tqdm(batch_indexes):
                da_batch = das_seqs[bi:bi + self.batch_size, :]
                text_batch = text_seqs[bi:bi + self.batch_size, :]
                bleu_batch = bleu_scores[bi:bi + self.batch_size, :]
                lp_batch = log_probs[bi:bi + self.batch_size, :]
                self.model.train_on_batch([text_batch, da_batch, lp_batch], bleu_batch)
                losses += self.model.evaluate([text_batch, da_batch, lp_batch], bleu_batch, batch_size=self.batch_size, verbose=0)
            train_loss = losses / das_seqs.shape[0] * self.batch_size
            time_spent = time() - start
            valid_loss = self.get_valid_loss(valid_das, valid_text_seqs, valid_log_probs, valid_bleu_scores)
            print('{} Epoch {} Train: {:.4f} Valid: {:.4f}'.format(time_spent, ep, train_loss, valid_loss))
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                epoch_since_minimum = 0
                self.save_model()
            else:
                epoch_since_minimum += 1
                if epoch_since_minimum > 5:
                    break
        self.load_model()

    def load_model(self):
        print("Loading trainable reranker from {}".format(self.save_location))
        model_path = os.path.join(self.save_location, "model.h5")
        if os.path.exists(model_path):
            # self.model = load_model(model_path)
            load_model_from_gpu(self.model, model_path)
        else:
            print("Model does not exist yet")

    def save_model(self):
        print("Saving trainable reranker at {}".format(self.save_location))
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        self.model.save(os.path.join(self.save_location, "model.h5"), save_format='h5')

    def predict_bleu_score(self, text_seqs, da_seqs, logprob_seqs):
        # need to normalise logprob_seqs
        logprob_seqs = (logprob_seqs - self.min_log_prob)/(self.max_log_prob-self.min_log_prob)
        return self.model.predict([text_seqs, da_seqs, logprob_seqs.reshape((1, 1))])


class TGEN_Reranker(object):
    def __init__(self, da_embedder, text_embedder, cfg_path):
        cfg = yaml.load(open(cfg_path, "r+"))
        self.batch_size = cfg['training_batch_size']
        self.lstm_size = cfg["hidden_size"]
        self.embedding_size = cfg['embedding_size']
        self.save_location = cfg["reranker_loc"]
        self.model = None
        self.text_embedder = text_embedder
        self.da_embedder = da_embedder
        self.set_up_models(text_embedder.length, text_embedder.vocab_length, da_embedder.inclusion_length)

    def set_up_models(self, len_in, vsize_in, len_out):
        lstm_type = CuDNNLSTM if is_gpu_available() else LSTM
        encoder_inputs = Input(shape=(len_in,), name='encoder_inputs')

        embed_enc = Embedding(input_dim=vsize_in, output_dim=self.embedding_size)
        encoder_lstm = lstm_type(self.lstm_size, return_sequences=True, return_state=True, name='encoder_lstm')
        en_lstm_out = encoder_lstm(embed_enc(encoder_inputs))
        h_n = en_lstm_out[1:]
        in_logistic_layer = Concatenate(axis=-1, name='concat_layer_Wy')(h_n)

        output = Dense(len_out, activation='sigmoid')(in_logistic_layer)

        self.model = Model(inputs=encoder_inputs, outputs=output)
        optimizer = Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy')
        self.model.summary()

    def get_valid_loss(self, valid_inc, valid_text):
        valid_loss = 0
        for bi in range(0, valid_inc.shape[0] - self.batch_size + 1, self.batch_size):
            valid_da_batch = valid_inc[bi:bi + self.batch_size, :]
            valid_text_batch = valid_text[bi:bi + self.batch_size, :]
            valid_loss += self.model.evaluate(valid_text_batch, valid_da_batch, batch_size=self.batch_size,
                                              verbose=0)
        return valid_loss

    def train(self, da_inclusion, text_seqs, epoch, valid_size, min_epoch=5):
        valid_inc = da_inclusion[-valid_size:]
        valid_text = text_seqs[-valid_size:]
        da_inclusion = da_inclusion[:-valid_size]
        text_seqs = text_seqs[:-valid_size]
        valid_losses = []
        min_valid_loss = math.inf
        epoch_since_last_min = 0
        for ep in range(epoch):
            start = time()
            losses = 0
            batch_indexes = list(range(0, da_inclusion.shape[0] - self.batch_size, self.batch_size))
            random.shuffle(batch_indexes)
            for bi in tqdm(batch_indexes):
                da_batch = da_inclusion[bi:bi + self.batch_size, :]
                text_batch = text_seqs[bi:bi + self.batch_size, :]
                self.model.train_on_batch(text_batch, da_batch)
                losses += self.model.evaluate(text_batch, da_batch, batch_size=self.batch_size, verbose=0)
            train_loss = losses / da_inclusion.shape[0] * self.batch_size

            valid_loss = self.get_valid_loss(valid_inc, valid_text)
            valid_losses.append(valid_loss)
            time_spent = time() - start
            print('{} Epoch {} Train: {:.4f} Valid {:.4f}'.format(time_spent, ep, train_loss, valid_loss))
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                epoch_since_last_min = 0
                self.save_model()
            else:
                epoch_since_last_min += 1
            if epoch_since_last_min == min_epoch:
                break
        self.load_model()
        final_valid_loss = self.get_valid_loss(valid_inc, valid_text)
        print("Final valid loss =", final_valid_loss)

    def predict(self, text_emb):
        preds = self.model.predict(text_emb)
        result = [[(1 if x > 0.5 else 0) for x in pred] for pred in preds]
        return result

    def get_pred_hamming_dist(self, text_emb, da_emb, da_embedder):
        pad = [self.text_embedder.tok_to_embed[PAD_TOK]] * (self.text_embedder.length - len(text_emb))
        pred = self.predict(np.array([pad + text_emb]))[0]
        das = da_embedder.reverse_embedding(da_emb)
        true = da_embedder.get_inclusion(das)
        return get_hamming_distance(pred, true)

    def load_model(self):
        print("Loading reranker from {}".format(self.save_location))
        self.model = load_model(os.path.join(self.save_location, "model.h5"))

    def save_model(self):
        print("Saving reranker at {}".format(self.save_location))
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        self.model.save(os.path.join(self.save_location, "model.h5"), save_format='h5')


class Regressor(object):
    def __init__(self, n_in, batch_size, max_len):
        self.batch_size = batch_size
        inputs = Input(batch_shape=(batch_size, n_in), name='encoder_inputs')
        dense1 = Dense(128, activation='relu')
        # dense3 = Dense(32, activation='relu')
        dense4 = Dense(1, activation='linear')
        self.model = Model(inputs=inputs, outputs=dense4(dense1(inputs)))
        optimizer = Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.max_len, self.min_len = max_len, 0
        self.max_lprob, self.min_lprob = 0, -100  # this value is approx from one experiment - may well change

    def normalise_features(self, f):
        f = np.array(f, copy=True)
        f[:, -1] = (f[:, -1] - self.min_len) / (self.max_len - self.min_len)
        f[:, -2] = (f[:, -2] - self.min_lprob) / (self.max_lprob - self.min_lprob)
        return f

    def train(self, features, labs):
        f = np.array(features)
        lens = f[:, -1]
        lprob = f[:, -2]
        self.max_len, self.min_len = max(lens), 0
        self.max_lprob, self.min_lprob = max(lprob), min(lprob)
        print("Min lprob: ", self.min_lprob)

        l = np.array(labs)
        # only care about order here:
        l = (l - l.min()) / (l.max() - l.min())
        f = self.normalise_features(f)
        print("Label Variation Mean {}, Var {}, Max {}, Min {}".format(l.mean(), l.var(), l.max(), l.min()))
        print("Start Train")
        self.model.fit(f, l, epochs=10, batch_size=1, verbose=0)
        print("End Train")

    def predict(self, features):
        f = self.normalise_features(features)
        return self.model.predict(f)

    def save_model(self, dir_name):
        self.model.save(os.path.join(dir_name, "classif.h5"), save_format='h5')

    def load_model(self, dir_name):
        self.model = load_model(os.path.join(dir_name, "classif.h5"))


class TGEN_Model(object):
    def __init__(self, da_embedder, text_embedder, cfg_path):
        cfg = yaml.load(open(cfg_path, "r+"))
        self.da_embedder = da_embedder
        self.text_embedder = text_embedder
        self.batch_size = cfg["train_batch_size"]
        self.vsize_out = text_embedder.vocab_length
        self.lstm_size = cfg["hidden_size"]
        self.embedding_size = cfg["embedding_size"]
        self.save_location = cfg["model_save_loc"]
        self.full_model = None
        self.encoder_model = None
        self.decoder_model = None
        self.set_up_models()
        self.greedy_complete_cache = {}
        self.greedy_complete_cache_path = cfg['greedy_complete_cache_location']

    def get_valid_loss(self, valid_da_seq, valid_text_seq, valid_onehot_seq):
        valid_loss = 0
        for bi in range(0, valid_da_seq.shape[0] - self.batch_size, self.batch_size):
            valid_da_batch = valid_da_seq[bi:bi + self.batch_size, :]
            valid_text_batch = valid_text_seq[bi:bi + self.batch_size, :]
            valid_onehot_batch = valid_onehot_seq[bi:bi + self.batch_size, :, :]
            valid_loss += self.full_model.evaluate([valid_da_batch, valid_text_batch[:, :-1]],
                                                   valid_onehot_batch[:, 1:, :],
                                                   batch_size=self.batch_size, verbose=0)
        return valid_loss

    def train(self, da_seq, text_seq, n_epochs, valid_size, early_stop_point=5, minimum_stop_point=20):
        valid_text_seq = text_seq[-valid_size:]
        valid_da_seq = da_seq[-valid_size:]
        text_seq = text_seq[:-valid_size]
        da_seq = da_seq[:-valid_size]
        valid_onehot_seq = to_categorical(valid_text_seq, num_classes=self.vsize_out)
        text_onehot_seq = to_categorical(text_seq, num_classes=self.vsize_out)

        valid_losses = []
        min_valid_loss = math.inf
        rev_embed = self.text_embedder.embed_to_tok
        print('\tValid Example:    {}'.format(" ".join([rev_embed[x] for x in valid_text_seq[0]]).replace('<>', '')))

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

            valid_loss = self.get_valid_loss(valid_da_seq, valid_text_seq, valid_onehot_seq)
            valid_losses.append(valid_loss)

            valid_pred = self.make_prediction(valid_da_seq[0], max_length=40).replace(
                "<> ", "")
            # train_pred = self.make_prediction(da_seq[0], text_embedder).replace("<>", "")
            time_taken = time() - start
            train_loss = losses / da_seq.shape[0] * self.batch_size
            valid_loss = valid_loss / valid_da_seq.shape[0] * self.batch_size

            print("({:.2f}s) Epoch {} Loss: {:.4f} Valid: {:.4f} {}".format(time_taken, ep + 1,
                                                                            train_loss, valid_loss,
                                                                            valid_pred))
            if valid_loss < min_valid_loss:
                self.save_model()
                min_valid_loss = valid_loss

            if len(valid_losses) - np.argmin(valid_losses) > early_stop_point and len(
                    valid_losses) > minimum_stop_point:
                return
        self.load_models()

        final_valid_loss = self.get_valid_loss(valid_da_seq, valid_text_seq, valid_onehot_seq)
        print("Final Valid Loss =", final_valid_loss)

    def load_models(self):
        print("Loading models at {}".format(self.save_location))
        load_model_from_gpu(self.full_model, os.path.join(self.save_location, "full.h5"))
        load_model_from_gpu(self.encoder_model, os.path.join(self.save_location, "enc.h5"))
        load_model_from_gpu(self.decoder_model, os.path.join(self.save_location, "dec.h5"))

    def save_model(self):
        print("Saving models at {}".format(self.save_location))
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        self.full_model.save(os.path.join(self.save_location, "full.h5"), save_format='h5')
        self.encoder_model.save(os.path.join(self.save_location, "enc.h5"), save_format='h5')
        self.decoder_model.save(os.path.join(self.save_location, "dec.h5"), save_format='h5')

    def beam_search_exapand(self, paths, enc_outs, beam_size):
        filled_paths = paths.copy()
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
        if len(out) == 3:
            dec_outs, dec_states = out[0], out[1:]
        else:  # old model
            dec_outs, dec_states = out[0], out[2:]
        new_paths = []
        tok_probs = []
        for p, dec_out, ds0, ds1 in zip(paths, dec_outs, dec_states[0], dec_states[1]):
            logprob, toks, ds = p
            if toks[-1] in self.text_embedder.end_embs:
                new_paths.append((logprob, toks, ds))
                continue
            top_k = np.argsort(dec_out, axis=-1)[0][-beam_size:]
            ds0 = ds0.reshape((1, -1))
            ds1 = ds1.reshape((1, -1))
            tok_prob = dec_out[0][top_k]
            for new_tok, tp in zip(top_k, tok_prob):
                tok_probs.append(tp)
                new_paths.append((logprob + log(tp), toks + [new_tok], [ds0, ds1]))
        return new_paths, tok_probs

    def make_prediction(self, encoder_in, beam_size=1, prev_tok=None, max_length=20):
        if prev_tok is None:
            prev_tok = START_TOK
        test_en = np.array([encoder_in])
        test_fr = [self.text_embedder.tok_to_embed[prev_tok]]

        inf_enc_out = self.encoder_model.predict(test_en)
        enc_outs = inf_enc_out[0]
        enc_last_state = inf_enc_out[1:]
        path = (log(1.0), test_fr, enc_last_state)
        return self.complete_search(path, enc_outs, beam_size, max_length)

    def complete_search(self, path, enc_outs, beam_size, max_length):
        paths = [path]
        for i in range(max_length):
            new_paths, _ = self.beam_search_exapand(paths, enc_outs, beam_size)
            paths = sorted(new_paths, key=lambda x: x[0], reverse=True)[:beam_size]
            if all([p[1][-1] in self.text_embedder.end_embs for p in paths]):
                break
        return " ".join(self.text_embedder.reverse_embedding(paths[0][1]))

    def save_cache(self):
        print("Saving cache")
        with open(self.greedy_complete_cache_path, 'wb+') as fp:
            msgpack.dump(self.greedy_complete_cache, fp)

    def populate_cache(self):
        print("Populating cache")
        if os.path.exists(self.greedy_complete_cache_path):
            with open(self.greedy_complete_cache_path, 'rb') as fp:
                self.greedy_complete_cache = msgpack.load(fp, use_list=False, strict_map_key=False)

    def complete_greedy(self, path, enc_outs, max_length):
        def get_key(max_length, path, enc_outs):
            return max_length, tuple(int(x) for x in path[1]), enc_outs

        key_enc_outs = tuple(float(x) for x in enc_outs.sum(axis=2).reshape(-1))
        so_far = []  # (key, next tok)
        paths = [path]
        completed = ""
        completed_logprob = 0
        for i in range(max_length):
            key = get_key(max_length - i, paths[0], key_enc_outs)
            if key in self.greedy_complete_cache:
                completed, completed_logprob = self.greedy_complete_cache[key]
                break
            else:
                paths, _ = self.beam_search_exapand(paths, enc_outs, 1)
                completed_logprob = paths[0][0]
                so_far.append((key, self.text_embedder.embed_to_tok[paths[0][1][-1]]))
                if all([p[1][-1] in self.text_embedder.end_embs for p in paths]):
                    break
        for key, tok in reversed(so_far):
            if completed:
                completed = tok + " " + completed
            else:
                completed = tok
            self.greedy_complete_cache[key] = (completed, completed_logprob)
        return completed, completed_logprob

    def set_up_models(self):
        len_in = self.da_embedder.length
        vsize_in = self.da_embedder.vocab_length
        len_out = self.text_embedder.length
        vsize_out = self.text_embedder.vocab_length
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
        dense_Ws = Dense(128, name='Ws')
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
        encoder_inf_inputs = Input(shape=(len_in), name='encoder_inf_inputs')
        en_lstm_out = encoder_lstm(embed_enc(encoder_inf_inputs))
        encoder_inf_out = en_lstm_out[0]
        encoder_inf_state = en_lstm_out[1:]

        self.encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_state])

        """ Decoder (Inference) model """
        dec_in = Input(shape=(1,), name='decoder_word_inputs')
        encoder_out = Input(shape=(len_in, self.lstm_size), name='encoder_inf_states')
        encoder_1 = Input(shape=(self.lstm_size,), name='decoder_init_1')
        encoder_2 = Input(shape=(self.lstm_size,), name='decoder_init_2')
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
