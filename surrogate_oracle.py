import argparse
import os

import msgpack
import numpy as np
import yaml

from base_models import TGEN_Model, TrainableReranker
from e2e_metrics.metrics.pymteval import BLEUScore
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from reimplement_reinforce import run_beam_search_with_rescorer
from scorer_functions import get_oracle_score_func, get_greedy_decode_score_func
from utils import get_training_variables, START_TOK, PAD_TOK, END_TOK


def save_scores_dict(i):
    print("Saving surrogate_train_dict after {} iterations".format(i))
    with open(cfg["surrogate_train_data_path"], 'wb+') as fp:
        msgpack.dump(surrogte_train_dict, fp)

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()

cfg_path = args.config_path
print("Using config from: {}".format(cfg_path))
cfg = yaml.load(open(cfg_path, "r"))
texts, das, = get_training_variables()
text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
da_embedder = DAEmbeddingSeq2SeqExtractor(das)

models = TGEN_Model(da_embedder, text_embedder, cfg)
models.load_models()

bleu_scorer = BLEUScore()
texts = [[[x for x in xs if x not in [START_TOK, END_TOK, PAD_TOK]]] for xs in texts]
final_scorer = get_oracle_score_func(bleu_scorer, texts, text_embedder, reverse=False)
if os.path.exists(cfg["surrogate_train_data_path"]) and cfg["load_surrogate_data"]:
    print("Loading Training data")
    surrogte_train_dict = msgpack.load(open(cfg["surrogate_train_data_path"], 'rb+'), use_list=False,
                                       strict_map_key=False)
else:
    print("Creating Training data")
    surrogte_train_dict = {}
    scorer_func = get_greedy_decode_score_func(models, final_scorer=final_scorer, max_length_out=text_embedder.length,
                                               save_scores=surrogte_train_dict)

    preds = run_beam_search_with_rescorer(scorer_func, models, das, 3, cfg['only_rerank_final'],
                                          cfg.get('beam_save_path', None), callback_1000=save_scores_dict)

    print("Saving training data")
    save_scores_dict(-1)

text_seqs = []
da_seqs = []
scores = []
valid_size = cfg['valid_size']
for (da_emb, text_emb), score in surrogte_train_dict.items():
    da_seqs.append(da_embedder.add_pad_to_embed(da_emb, to_start=True))
    text_seqs.append(text_embedder.add_pad_to_embed(text_emb, to_start=True))
    scores.append(score)

text_seqs = np.array(text_seqs)
da_seqs = np.array(da_seqs)
scores = np.array(scores).reshape((-1, 1))
reranker = TrainableReranker(da_embedder, text_embedder, cfg)
reranker.train(text_seqs[:-valid_size], da_seqs[:-valid_size], scores[:-valid_size], cfg["trainable_reranker_epoch"],
               text_seqs[-valid_size:], da_seqs[-valid_size:], scores[-valid_size:])
