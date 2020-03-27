# ./run_tgen.py seq2seq_train
import os
import sys
from time import time

import torch
from gensim.models import Word2Vec
from torch import nn
from tqdm import tqdm

from e2e_metrics.metrics.pymteval import BLEUScore

sys.path.append(os.path.join(os.getcwd(), 'tgen'))
from utils import construct_logs, RERANK, get_truth_training
from getopt import getopt

from beam_search_edit import _beam_search, lexicalize_beam, save_training_data, rolling_beam_search, _init_beam_search
from e2e_metrics.measure_scores import load_data, evaluate, run_pymteval

import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
from tgen.tree import TreeData
from tgen.tfclassif import Reranker
from tgen.config import Config
from tgen.seq2seq import Seq2SeqGen, Seq2SeqBase, cut_batch_into_steps
from tgen.logf import log_info, log_debug, log_warn
from tgen.futil import read_das, write_ttrees, write_tokens, postprocess_tokens, create_ttree_doc, smart_load_absts

import tensorflow as tf

true_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/tgen/e2e-challenge/input/devel-text.txt"

da_train_file = "tgen/e2e-challenge/input/train-das.txt"
text_train_file = "tgen/e2e-challenge/input/train-text.txt"
seq2seq_config_file = "tgen/e2e-challenge/config/config.yaml"
da_test_file = "tgen/e2e-challenge/input/devel-das.txt"
test_abstr_file = "tgen/e2e-challenge/input/devel-abst.txt"
train_abstr_file = "tgen/e2e-challenge/input/train-abst.txt"

ngpu = 0


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def initialise(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    else:
        model.apply(weights_init)
    print(model)


class BinClassifier(nn.Module):
    def __init__(self, ngpu, n_in):
        super(BinClassifier, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def safe_get_w2v(w2v, tok):
    unimp_toks = ['<VOID>', '<UNK>', '<-s>']
    tok = '<STOP>' if tok in unimp_toks else tok
    return w2v[tok]


def get_features(path, tgen, w2v):
    hidden_state = list(path.dec_states[0].c[0])
    out_state = list(path.dec_states[0].h[0])
    pred_words = tgen.tree_embs.ids_to_strings([inp[0] for inp in path.dec_inputs])

    return np.concatenate((hidden_state,
                           out_state,
                           safe_get_w2v(w2v, pred_words[-1]),
                           safe_get_w2v(w2v, pred_words[-2]),
                           [path.logprob]))


def reinforce_learning(beam_size, epoch, seq2seq_model_file="models/model_e2e_2/model.pickle.gz"):
    tf.reset_default_graph()
    w2v = Word2Vec.load("models/word2vec_30.model")

    tgen = Seq2SeqBase.load_from_file(seq2seq_model_file)
    tgen.beam_size = beam_size

    das = read_das(da_train_file)
    truth = [[x] for x in get_truth_training()]
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    D = []
    n_in = 161
    classif_path = "models/classif_{}.model".format(n_in)
    classifier = BinClassifier(ngpu, n_in).to(device)
    classifier = nn.DataParallel(classifier, list(range(ngpu)))
    initialise(classifier, classif_path)
    # might be good to initialise with features
    for i in range(epoch):
        for d, (da, true) in tqdm(enumerate(zip(das, truth))):
            tgen_time_spent = 0
            enc_inputs = np.array([[x] for x in tgen.da_embs.get_embeddings(da)])
            _init_beam_search(tgen, enc_inputs)
            empty_tree_emb = tgen.tree_embs.get_embeddings(TreeData())
            dec_inputs = np.array([[x] for x in empty_tree_emb])
            true_toks = true[0].split(" ")
            paths = [tgen.DecodingPath(stop_token_id=tgen.tree_embs.STOP, dec_inputs=[dec_inputs[0]])]
            for step in range(len(true_toks)):
                # print(step, " ***********")
                new_paths = []
                # expand
                start = time()
                for path in paths:
                    out_probs, st = tgen._beam_search_step(path.dec_inputs, path.dec_states)
                    new_paths.extend(path.expand(beam_size, out_probs, st))
                tgen_time_spent += time() - start
                # prune and create data
                path_scores = []
                for path in new_paths:
                    features = get_features(path, tgen, w2v)

                    # get score classif
                    t_features = torch.Tensor(features)

                    classif_score = classifier(t_features).detach()[0].item()
                    path_scores.append((classif_score, path))

                    # get score reference
                    # greedy decode
                    start = time()
                    while path.dec_inputs[-1] not in [tgen.tree_embs.VOID, tgen.tree_embs.STOP] \
                            and len(path.dec_states) < len(path.dec_inputs):
                        out_probs, st = tgen._beam_search_step(path.dec_inputs, path.dec_states)
                        path = path.expand(1, out_probs, st)[0]
                    tgen_time_spent += time() - start
                    out_sent = tgen.tree_embs.ids_to_strings([inp[0] for inp in path.dec_inputs])
                    out_sent = [x for x in out_sent if x not in ['<GO>', '<STOP>', '<VOID>']]
                    # print(" ".join(out_sent))
                    # This might be a poor way of scoring as greedy decode can get low bleu scores
                    bleu = BLEUScore()
                    bleu.append(out_sent, [true_toks])
                    ref_score = bleu.score()
                    D.append((features, ref_score))
                paths = [x[1] for x in sorted(path_scores, key=lambda y: y[0], reverse=True)[:beam_size]]

                if all([p.dec_inputs[-1] in [tgen.tree_embs.VOID, tgen.tree_embs.STOP] for p in paths]):
                    break  # stop decoding if we have reached the end in all paths
            # print(tgen_time_spent)

            # Train on D


if __name__ == "__main__":
    reinforce_learning(3, 5)
