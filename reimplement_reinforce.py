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
from reinforcement_learning import safe_get_w2v
from utils import get_texts_training, RERANK

sys.path.append(os.path.join(os.getcwd(), 'tgen'))
from tgen.futil import read_das

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import LSTM, TimeDistributed, Dense, Concatenate, Input, Embedding, CuDNNLSTM


class BinClassifier(object):
    def __init__(self, n_in, batch_size):
        self.batch_size = batch_size
        inputs = Input(batch_shape=(batch_size, n_in), name='encoder_inputs')
        dense1 = Dense(256, activation='relu')
        dense2 = Dense(128, activation='relu')
        dense3 = Dense(32, activation='relu')
        dense4 = Dense(1, activation='sigmoid')
        self.model = Model(inputs=inputs, outputs=dense4(dense3(dense2(dense1(inputs)))))
        optimizer = Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def train(self, features, lables, n_epoch):
        for ep in range(n_epoch):
            start = time()
            losses = 0
            batch_indexes = list(range(0, features.shape[0] - self.batch_size, self.batch_size))
            random.shuffle(batch_indexes)
            for bi in batch_indexes:
                feature_batch = features[bi:bi + self.batch_size, :]
                lab_batch = lables[bi:bi + self.batch_size]
                self.model.train_on_batch([feature_batch], lab_batch)
                losses += self.model.evaluate([feature_batch], lab_batch, batch_size=self.batch_size, verbose=0)
            if (ep + 1) % 1 == 0:
                time_taken = time() - start
                train_loss = losses
                print("({:.2f}s) Epoch {} Loss: {:.4f}".format(time_taken, ep + 1, train_loss))

    def predict(self, features):
        return self.model.predict(features)

    def save_model(self,dir_name):
        self.model.save(os.path.join(dir_name, "classif.h5"), save_format='h5')



def get_features(path, text_embedder, w2v):
    h = path[2][0][0]
    c = path[2][1][0]
    pred_words = [text_embedder.embed_to_tok[x] for x in path[1]]

    return np.concatenate((h, c,
                           safe_get_w2v(w2v, pred_words[-1]), safe_get_w2v(w2v, pred_words[-2]),
                           [path[0]]))


def safe_get_w2v(w2v, tok):
    unimp_toks = ['<>']
    tok = '<E>' if tok in unimp_toks else tok
    return w2v[tok]


def load_rein_data(filepath):
    with open(filepath, "r") as fp:
        features = []
        labs = []
        for line in fp.readlines():
            line = [float(x) for x in line.split(",")]
            labs.append(line[-1])
            features.append(line[:-1])
        return features, labs


def train_classifier(classifier, features, labs):
    f = np.array(features)
    l = np.array(labs)
    classifier.model.fit(f, l, epochs=10, batch_size=1, verbose=2)


def reinforce_learning(beam_size, epoch, data_save_path, beam_search_model: TGEN_Model, das, truth,
                       chance_of_choosing=0.01):
    w2v = Word2Vec.load("models/word2vec_30.model")

    D = []
    n_in = 317
    batch_size = 1
    classifier = BinClassifier(n_in, batch_size=1)
    if os.path.exists("reinforce_data.txt"):
        feats, labs = load_rein_data(data_save_path)
        train_classifier(classifier, feats, labs)
    bleu = BLEUScore()
    bleu_overall = BLEUScore()
    # might be good to initialise with features
    data_save_file = open(data_save_path, "a+")
    for i in range(epoch):
        for i,(da_emb, true) in tqdm(enumerate(zip(da_embedder.get_embeddings(das), truth))):
            test_en = np.array([da_emb])
            test_fr = [text_embedder.tok_to_embed['<S>']]
            inf_enc_out = beam_search_model.encoder_model.predict(test_en)
            enc_outs = inf_enc_out[0]
            enc_last_state = inf_enc_out[1:]
            paths = [(log(1.0), test_fr, enc_last_state)]
            end_tokens = [text_embedder.tok_to_embed['<E>'], text_embedder.tok_to_embed['<>']]

            for step in range(len(true)):
                new_paths = beam_search_model.beam_search_exapand(paths, end_tokens, enc_outs, beam_size)

                path_scores = []
                for path in new_paths:
                    features = get_features(path, text_embedder, w2v)
                    classif_score = classifier.predict(features.reshape(1, -1))
                    path_scores.append((classif_score, path))

                    # get score reference
                    # greedy decode
                    if random.random() < chance_of_choosing:
                        rest = beam_search_model.make_prediction(da_emb, text_embedder,
                                                                 beam_size=1,
                                                                 prev_tok=text_embedder.embed_to_tok[path[1][-1]],
                                                                 max_length=5 - len(path[1]))
                        cur = " ".join([text_embedder.embed_to_tok[x] for x in path[1]])

                        # This might be a poor way of scoring as greedy decode can get low bleu scores
                        bleu.reset()
                        bleu.append(cur + " " + rest, [true])
                        ref_score = bleu.score()
                        D.append((features, ref_score))
                        data_save_file.write("{},{}\n".format(",".join([str(x) for x in features]), ref_score))

                paths = [x[1] for x in sorted(path_scores, key=lambda y: y[0], reverse=True)[:beam_size]]
                if all([p[1][-1] in end_tokens for p in paths]):
                    bleu_overall.append(paths[-1],[true])
                    break

            if i % 1000 == 1 and i > 100:
                score = bleu_overall.score()
                bleu_overall.reset()
                print("BLEU SCORE FOR last batch = {}".format(score))
                features = [d[0] for d in D]
                labs = [d[1] for d in D]
                train_classifier(classifier, features, labs)
                model_save_loc = "models/reimplementation"
                classifier.save_model(model_save_loc)


cfg = yaml.load(open("config.yaml", "r"))
use_size = cfg['use_size']
valid_size = cfg['valid_size']
epoch = cfg['epoch']
batch_size = cfg['batch_size']
hidden_size = cfg['hidden_size']
embedding_size = cfg['embedding_size']
load_from_save = cfg['load_from_save']
model_save_loc = "models/reimplementation"

das = read_das("tgen/e2e-challenge/input/train-das.txt")

texts = [['<S>'] + x + ['<E>'] for x in get_texts_training()]
print(texts[0])

text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
da_embedder = DAEmbeddingSeq2SeqExtractor(das)

das = das[:use_size + valid_size]
texts = texts[:use_size + valid_size]

text_embs = text_embedder.get_embeddings(texts)
text_vsize = text_embedder.vocab_length
text_len = len(text_embs[0])

da_embs = da_embedder.get_embeddings(das)
da_vsize = da_embedder.vocab_length
da_len = len(da_embs[0])
print(da_vsize, text_vsize, da_len, text_len)

# train_in = np.array(da_embs)
# train_out = np.array(text_embs)

beam_size = 3
models = TGEN_Model(batch_size, da_len, text_len, da_vsize, text_vsize, hidden_size, embedding_size, beam_size)
models.load_models_from_location(model_save_loc)

reinforce_learning(beam_size, 4, "output_files/training_data/{}.csv".format(beam_size), models, das, texts)
