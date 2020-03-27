import os
import pickle
import random
import sys
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

from utils import get_truth_training, count_lines

sys.path.append(os.path.join(os.getcwd(), 'tgen'))
from tgen.seq2seq import Seq2SeqBase

from_file = "output_files/training_data/5_v3_big.csv"
to_file = "output_files/training_data/5_v6_big.feat"
if not os.path.exists(to_file):
    open(to_file, "w+")

end = count_lines(from_file)
start = count_lines(to_file)
batch_size = 10000
while start < end:
    training_data = []
    print("Start =", start)
    print("Loading from file")
    with open(from_file, "r") as fp:
        for i, line in tqdm(enumerate(fp.readlines())):
            if i < start:
                continue
            elif i > start + batch_size:
                break
            lstm_feat, rest = line.strip('\n').strip("[").split("]")
            lstm_feat = [float(x) for x in lstm_feat.split(',')]
            rest_split = rest.strip(',').split(',')
            id = int(rest_split[0])
            lab = int(rest_split[-1])
            log_prob = float(rest_split[-2])
            sent = ",".join(rest_split[1:-2])
            toks = sent.split(" ")
            training_data.append((lstm_feat, id, toks, log_prob, lab))

    data_points = []
    w2v = Word2Vec.load("models/word2vec.model")
    print("Processsing")
    for t in tqdm(training_data):
        pred_toks = t[2]
        pred_sent = " ".join(pred_toks[1:]).replace(" <STOP>", "")
        tok = pred_toks[-1]
        prev_tok = pred_toks[-2]
        # Hacky solution for now
        unimp_toks = ['<VOID>', '<UNK>', '<-s>']
        tok = '<STOP>' if tok in unimp_toks else tok
        prev_tok = '<STOP>' if prev_tok in unimp_toks else prev_tok

        embed = w2v.wv[tok]
        prev_embed = w2v.wv[prev_tok]
        data_points.append(t[0] + list(embed) + list(prev_embed) + [t[3], t[4]])

    print("Writing to file")
    with open(to_file, "a+") as fp:
        for dp in tqdm(data_points):
            fp.write(','.join([str(x) for x in dp]) + '\n')
    start += batch_size
