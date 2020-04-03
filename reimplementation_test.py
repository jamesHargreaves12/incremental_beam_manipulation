import random
import sys
import os
from time import time
import numpy as np
import yaml
from tqdm import tqdm

from base_model import TGEN_Model
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from utils import get_texts_training, RERANK, apply_absts

sys.path.append(os.path.join(os.getcwd(), 'tgen'))
from tgen.futil import read_das, smart_load_absts

cfg = yaml.load(open("config_train.yaml", "r"))
use_size = cfg['use_size']
valid_size = cfg['valid_size']
epoch = cfg['epoch']
batch_size = cfg['batch_size']
hidden_size = cfg['hidden_size']
embedding_size = cfg['embedding_size']
load_from_save = cfg['load_from_save']

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

train_in = np.array(da_embs)
train_out = np.array(text_embs)

if load_from_save and os.path.exists(cfg['model_save_loc']):
    models = TGEN_Model(da_len, text_len, da_vsize, text_vsize, 3, cfg)
    models.load_models_from_location(cfg['model_save_loc'])
else:
    models = TGEN_Model(da_len, text_len, da_vsize, text_vsize, 1, cfg)
    models.train(train_in[:-valid_size], train_out[:-valid_size], epoch, train_in[-valid_size:],
                 train_out[-valid_size:], text_embedder)
    models.save_model(cfg['model_save_loc'])

print("TESTING")
test_das = read_das("tgen/e2e-challenge/input/devel-das.txt")
absts = smart_load_absts('tgen/e2e-challenge/input/train-abst.txt')
for beam_size in [1, 5, 10, 30, 100]:
    print("Beam_size {}".format(beam_size))
    start = time()
    results = []
    for da_emb in tqdm(da_embedder.get_embeddings(test_das)):
        pred = models.make_prediction(da_emb, text_embedder, beam_size, max_length=text_len)
        results.append(pred.replace(" <>", "").replace('<S> ','').replace(' <E>','').split(" "))
    post_abstr = apply_absts(absts, results)
    with open(cfg["res_save_format"].format(beam_size), "w+") as out_file:
        for pa in post_abstr:
            out_file.write(" ".join(pa) + '\n')
    print(time() - start)