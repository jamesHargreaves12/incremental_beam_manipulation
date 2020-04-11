import random
import sys
import os
from time import time
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from base_model import TGEN_Model, TGEN_Reranker
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from reimplement_reinforce import get_tgen_rerank_score_func, run_beam_search_with_rescorer
from utils import get_texts_training, RERANK, apply_absts, get_hamming_distance

sys.path.append(os.path.join(os.getcwd(), 'tgen'))
from tgen.futil import read_das, smart_load_absts

cfg = yaml.load(open("configs/config_train.yaml", "r"))
use_size = cfg['use_size']
valid_size = cfg['valid_size']
epoch = cfg['epoch']
batch_size = cfg['batch_size']
hidden_size = cfg['hidden_size']
embedding_size = cfg['embedding_size']
load_from_save = cfg['load_from_save']
train_reranker = True
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

train_da = np.array(da_embs)
train_text = np.array(text_embs)

if train_reranker:
    reranker_loc = "models/reranker/tgen"
    load_reranker = False
    plot_reranker_stats = True

    train_text = np.array(text_embedder.get_embeddings(texts, pad_from_end=False)+ [text_embedder.empty_embedding])
    das_inclusions = np.array([da_embedder.get_inclusion(da) for da in das] + [da_embedder.empty_inclusion])
    reranker = TGEN_Reranker(text_len, text_vsize, da_embedder.inclusion_length, text_embedder, cfg)
    if os.path.exists(reranker_loc) and load_reranker:
        reranker.load_models_from_location(reranker_loc)
    else:
        reranker.train(das_inclusions[:-valid_size], train_text[:-valid_size], 14, das_inclusions[-valid_size:],
                       train_text[-valid_size:])
        reranker.save_model(reranker_loc)
    if plot_reranker_stats:
        preds = reranker.predict(train_text)
        ham_dists = [get_hamming_distance(x, y) for x, y in zip(preds, das_inclusions)]
        filter_hams = [x for x in ham_dists if x != 0]
        plt.hist(filter_hams)
        plt.show()

exit()
if load_from_save and os.path.exists(cfg['model_save_loc']):
    models = TGEN_Model(da_len, text_len, da_vsize, text_vsize, 3, cfg)
    models.load_models_from_location(cfg['model_save_loc'])
else:
    models = TGEN_Model(da_len, text_len, da_vsize, text_vsize, 1, cfg)
    models.train(train_da[:-valid_size], train_text[:-valid_size], epoch, train_da[-valid_size:],
                 train_text[-valid_size:], text_embedder)
    models.save_model(cfg['model_save_loc'])

print("TESTING")
test_das = read_das("tgen/e2e-challenge/input/devel-das.txt")
absts = smart_load_absts('tgen/e2e-challenge/input/train-abst.txt')
for beam_size in [3, 5, 10, 30, 100]:
    print("Beam_size {}".format(beam_size))
    start = time()
    results = []
    for da_emb in tqdm(da_embedder.get_embeddings(test_das)):
        pred = models.make_prediction(da_emb, text_embedder, beam_size, max_length=text_len)
        results.append(pred.replace(" <>", "").replace('<S> ', '').replace(' <E>', '').split(" "))
    post_abstr = apply_absts(absts, results)
    with open(cfg["res_save_format"].format(beam_size), "w+") as out_file:
        for pa in post_abstr:
            out_file.write(" ".join(pa) + '\n')
    print(time() - start)
