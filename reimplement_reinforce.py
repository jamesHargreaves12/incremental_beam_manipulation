from math import log

from gensim.models import Word2Vec
import random
import sys
import os
from time import time
import numpy as np
import yaml
import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import RMSprop
from tqdm import tqdm

from base_model import TGEN_Model
from e2e_metrics.metrics.pymteval import BLEUScore
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from utils import get_texts_training, RERANK, get_training_das_texts, safe_get_w2v, apply_absts

sys.path.append(os.path.join(os.getcwd(), 'tgen'))
from tgen.futil import read_das, smart_load_absts

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import LSTM, TimeDistributed, Dense, Concatenate, Input, Embedding, CuDNNLSTM


class BinClassifier(object):
    def __init__(self, n_in, batch_size, max_len):
        self.batch_size = batch_size
        inputs = Input(batch_shape=(batch_size, n_in), name='encoder_inputs')
        dense1 = Dense(128, activation='relu')
        # dense3 = Dense(32, activation='relu')
        dense4 = Dense(1, activation='sigmoid')
        self.model = Model(inputs=inputs, outputs=dense4(dense1(inputs)))
        optimizer = Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.max_len, self.min_len = max_len, 0
        self.max_lprob, self.min_lprob = 0, -100 #this value is approx from one experiment - may well change

    # def train(self, features, lables, n_epoch):
    #     for ep in range(n_epoch):
    #         start = time()
    #         losses = 0
    #         batch_indexes = list(range(0, features.shape[0] - self.batch_size, self.batch_size))
    #         random.shuffle(batch_indexes)
    #         for bi in batch_indexes:
    #             feature_batch = features[bi:bi + self.batch_size, :]
    #             lab_batch = lables[bi:bi + self.batch_size]
    #             self.model.train_on_batch([feature_batch], lab_batch)
    #             losses += self.model.evaluate([feature_batch], lab_batch, batch_size=self.batch_size, verbose=0)
    #         if (ep + 1) % 1 == 0:
    #             time_taken = time() - start
    #             train_loss = losses
    #             print("({:.2f}s) Epoch {} Loss: {:.4f}".format(time_taken, ep + 1, train_loss))
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
        f = self.normalise_features(f)
        print("Label Variation Mean {}, Var {}, Max {}, Min {}".format(l.mean(), l.var(), l.max(), l.min()))
        self.model.fit(f, l, epochs=10, batch_size=1, verbose=2)

    def predict(self, features):
        f = self.normalise_features(features)
        return self.model.predict(f)

    def save_model(self, dir_name):
        self.model.save(os.path.join(dir_name, "classif.h5"), save_format='h5')

    def load_model(self, dir_name):
        self.model = load_model(os.path.join(dir_name, "classif.h5"))


def get_features(path, text_embedder, w2v, tok_prob):
    h = path[2][0][0]
    c = path[2][1][0]
    pred_words = [text_embedder.embed_to_tok[x] for x in path[1]]

    return np.concatenate((h, c,
                           safe_get_w2v(w2v, pred_words[-1]), safe_get_w2v(w2v, pred_words[-2]), [tok_prob, path[0], len(pred_words)]))


def load_rein_data(filepath):
    with open(filepath, "r") as fp:
        features = []
        labs = []
        for line in fp.readlines():
            line = [float(x) for x in line.split(",")]
            labs.append(line[-1])
            features.append(line[:-1])
        return features, labs


def get_completion_score(beam_search_model, da_emb, path, bleu, true, text_embedder):
    cur = " ".join(text_embedder.reverse_embedding(path[1]))
    rest = beam_search_model.make_prediction(da_emb, text_embedder,
                                             beam_size=1,
                                             prev_tok=text_embedder.embed_to_tok[path[1][-1]],
                                             max_length=len(true) - len(path[1]))
    bleu.reset()
    bleu.append(cur + " " + rest, [true])
    return bleu.score()


def reinforce_learning(beam_size, data_save_path, beam_search_model: TGEN_Model, das, truth, classifier, text_embedder,
                       da_embedder, cfg,
                       chance_of_choosing=0.01):
    w2v = Word2Vec.load(cfg["w2v_path"])
    D = []
    bleu = BLEUScore()
    bleu_overall = BLEUScore()
    beam_search_proportion = 1.0
    bsp_multiplier = cfg['reference_policy_decay_rate']

    data_save_file = open(data_save_path, "a+")
    for i in range(cfg["epoch"]):
        beam_search_proportion *= bsp_multiplier
        for j, (da_emb, true) in tqdm(enumerate(zip(da_embedder.get_embeddings(das), truth))):
            inf_enc_out = beam_search_model.encoder_model.predict(np.array([da_emb]))
            enc_outs = inf_enc_out[0]
            enc_last_state = inf_enc_out[1:]
            paths = [(log(1.0), text_embedder.start_emb, enc_last_state)]
            end_tokens = text_embedder.end_embs

            for step in range(len(true)):
                new_paths, tok_probs = beam_search_model.beam_search_exapand(paths, end_tokens, enc_outs, beam_size)

                path_scores = []
                classifier_order = random.random() > beam_search_proportion
                for path, tp in zip(new_paths, tok_probs):
                    features = get_features(path, text_embedder, w2v, tp)
                    if classifier_order:
                        classif_score = classifier.predict(features.reshape(1, -1))
                        path_scores.append((classif_score, path))
                    else:
                        path_scores.append((path[0] + log(tp), path))

                    # greedy decode
                    if random.random() < chance_of_choosing:
                        ref_score = get_completion_score(beam_search_model, da_emb, path, bleu, true, text_embedder)
                        D.append((features, ref_score))
                        data_save_file.write("{},{}\n".format(",".join([str(x) for x in features]), ref_score))
                # prune
                paths = [x[1] for x in sorted(path_scores, key=lambda y: y[0], reverse=True)[:beam_size]]

                if all([p[1][-1] in end_tokens for p in paths]):
                    pred = text_embedder.reverse_embedding(paths[0][1])
                    bleu_overall.append(pred, [true])
                    break

        score = bleu_overall.score()
        bleu_overall.reset()
        print("BLEU SCORE FOR last batch = {}".format(score))
        features = [d[0] for d in D]
        labs = [d[1] for d in D]
        classifier.train(features, labs)
        classifier.save_model(cfg["model_save_loc"])
        test_res = run_classifier_bs(classifier, beam_search_model, None, None, text_embedder, da_embedder,
                                     das[:1], beam_size, cfg)
        print(" ".join(test_res[0]))


def run_classifier_bs(classifier, beam_search_model, out_path, abstss, text_embedder, da_embedder, das, beam_size, cfg):
    w2v = Word2Vec.load(cfg["w2v_path"])
    max_predict_len = 20

    results = []
    for i, da_emb in tqdm(enumerate(da_embedder.get_embeddings(das))):
        inf_enc_out = beam_search_model.encoder_model.predict(np.array([da_emb]))
        enc_outs = inf_enc_out[0]
        enc_last_state = inf_enc_out[1:]
        paths = [(log(1.0), text_embedder.start_emb, enc_last_state)]
        end_tokens = text_embedder.end_embs

        for step in range(max_predict_len):
            new_paths, tok_probs = beam_search_model.beam_search_exapand(paths, end_tokens, enc_outs, beam_size)

            path_scores = []
            for path, tp in zip(new_paths, tok_probs):
                features = get_features(path, text_embedder, w2v, tp)
                classif_score = classifier.predict(features.reshape(1, -1))
                path_scores.append((classif_score, path))
            # prune
            paths = [x[1] for x in sorted(path_scores, key=lambda y: y[0], reverse=True)[:beam_size]]

            if all([p[1][-1] in end_tokens for p in paths]):
                break
        best_path = paths[0]
        pred_toks = text_embedder.reverse_embedding(best_path[1])
        results.append(pred_toks)
    if out_path:
        post_abstr = apply_absts(abstss, results)
        with open(out_path, "w+") as out_file:
            for pa in post_abstr:
                out_file.write(" ".join(pa) + '\n')
    else:
        return results


if __name__ == "__main__":
    beam_size = 3
    cfg = yaml.load(open("config_reinforce.yaml", "r"))
    train_data_location = "output_files/training_data/{}_.csv".format(beam_size)
    das, texts = get_training_das_texts()
    text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
    da_embedder = DAEmbeddingSeq2SeqExtractor(das)
    das = das[:cfg['use_size']]
    texts = texts[:cfg['use_size']]
    n_in = cfg["hidden_size"] * 2 + cfg["w2v_size"] * 2 + 3

    text_vsize, text_len = text_embedder.vocab_length, text_embedder.length
    da_vsize, da_len = da_embedder.vocab_length, da_embedder.length
    print(da_vsize, text_vsize, da_len, text_len)

    models = TGEN_Model(da_len, text_len, da_vsize, text_vsize, beam_size, cfg)
    models.load_models_from_location(cfg["model_save_loc"])

    classifier = BinClassifier(n_in, batch_size=1, max_len=max([len(x) for x in texts]))
    if cfg["classif_from_file"]:
        classifier.load_model(cfg["model_save_loc"])
    elif os.path.exists(train_data_location):
        feats, labs = load_rein_data(train_data_location)
        if feats:
            classifier.train(feats, labs)

    reinforce_learning(beam_size, train_data_location, models, das, texts, classifier, text_embedder, da_embedder, cfg)
    # save_path = "output_files/out-text-dir-v2/rein_{}.txt".format(beam_size)
    # absts = smart_load_absts('tgen/e2e-challenge/input/train-abst.txt')
    # run_classifier_bs(classifier, models, save_path, absts, text_embedder, da_embedder, das, beam_size, cfg)
