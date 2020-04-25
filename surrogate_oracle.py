import argparse
import os

import msgpack
import numpy as np
import yaml
import matplotlib.pyplot as plt

from utils import get_training_variables, START_TOK, PAD_TOK, END_TOK, get_multi_reference_training_variables, \
    get_final_beam, get_test_das, get_true_sents
from base_models import TGEN_Model, TrainableReranker
from e2e_metrics.metrics.pymteval import BLEUScore
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from reimplement_reinforce import run_beam_search_with_rescorer
from scorer_functions import get_oracle_score_func, get_greedy_decode_score_func, get_score_function


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
texts, das, = get_multi_reference_training_variables()
da_embedder = DAEmbeddingSeq2SeqExtractor(das)
# This is a very lazy move
texts_flat, _ = get_training_variables()
text_embedder = TokEmbeddingSeq2SeqExtractor(texts_flat)

models = TGEN_Model(da_embedder, text_embedder, cfg["tgen_seq2seq_config"])
models.load_models()

bleu_scorer = BLEUScore()
texts = [[x for x in xs if x not in [START_TOK, END_TOK, PAD_TOK]] for xs in texts]
final_scorer = get_oracle_score_func(bleu_scorer, texts, text_embedder, reverse=False)
should_load_data = os.path.exists(cfg["surrogate_train_data_path"]) and cfg["load_surrogate_data"]
surrogte_train_dict = {}

models.populate_cache()

if should_load_data:
    print("Loading Training data")
    surrogte_train_dict = msgpack.load(open(cfg["surrogate_train_data_path"], 'rb+'), use_list=False,
                                       strict_map_key=False)
if not should_load_data or cfg["get_rest_surrogate_data"]:
    print("Creating Training data")
    scorer_func = get_greedy_decode_score_func(models, final_scorer=final_scorer, max_length_out=text_embedder.length,
                                               save_scores=surrogte_train_dict)
    start_point = cfg.get("surrogate_data_start_point", 0)
    preds = run_beam_search_with_rescorer(scorer=scorer_func,
                                          beam_search_model=models,
                                          das=das[start_point:],
                                          beam_size=3,
                                          only_rerank_final=False,
                                          save_final_beam_path=cfg.get('beam_save_path', None),
                                          callback_1000=save_scores_dict)

    print("Saving training data")
    save_scores_dict(-1)
    models.save_cache()

print("Training")
text_seqs = []
da_seqs = []
scores = []
log_probs = []
print(len(surrogte_train_dict))
for (da_emb, text_emb), (score, log_prob) in surrogte_train_dict.items():
    da_seqs.append(da_embedder.add_pad_to_embed(da_emb, to_start=True))
    text_seqs.append(text_embedder.add_pad_to_embed(text_emb, to_start=True))
    scores.append(score)
    log_probs.append(log_prob)

valid_size = cfg['valid_size']
text_seqs = np.array(text_seqs)
da_seqs = np.array(da_seqs)
scores = np.array(scores).reshape((-1, 1))
log_probs = np.array(log_probs).reshape((-1, 1))

# log probs need to be normalised
print("Before: ", np.min(log_probs), np.ptp(log_probs))
log_probs = (log_probs - np.min(log_probs)) / np.ptp(log_probs)
print("After: ", np.min(log_probs), np.ptp(log_probs))
if cfg['renormalise_scores']:
    orig_mean = scores.mean()
    orig_sd = scores.std()
    new_mean = 0.5
    new_sd = 0.4 # clip range is approx 0.45-> 0.9 (20% of data is clipped)
    new_scores = (scores - orig_mean)/orig_sd*new_sd + new_mean
    new_scores = new_scores.clip(0, 1)
    print("(μ,σ) = ({},{}) -> ({},{})".format(orig_mean, orig_sd, new_scores.mean(), new_scores.std()))
    scores = new_scores

reranker = TrainableReranker(da_embedder, text_embedder, cfg_path)
reranker.load_model()

if "train" in cfg and cfg["train"]:
    reranker.train(text_seqs, da_seqs, scores, log_probs, cfg["epoch"], valid_size, cfg.get("min_passes", 5))

if "get_stats" in cfg and cfg["get_stats"]:
    test_das = get_test_das()
    test_texts = get_true_sents()
    # print("Loading final beams")
    # scorer = get_score_function('identity', cfg, models, None)
    # run_beam_search_with_rescorer(scorer, models, test_das, 3, only_rerank_final=False,
    #                               save_final_beam_path='output_files/saved_beams/vanilla_3.txt')

    test_da_embs = da_embedder.get_embeddings(test_das)
    final_beam = get_final_beam(3)
    beam_texts = [[text for text, _ in beam] for beam in final_beam]
    beam_tok_logprob = [[tp for _, tp in beam] for beam in final_beam]
    # test_text_embs = [text_embedder.get_embeddings(beam) for beam in beam_texts]
    bleu = BLEUScore()
    mapping = []
    for texts, da_emb, tp_emb, true_texts in zip(beam_texts, test_da_embs, beam_tok_logprob, test_texts):
        text_seqs = np.array(text_embedder.get_embeddings(texts, pad_from_end=False))
        da_seqs = np.array([da_emb for _ in range(len(text_seqs))])
        tp_seqs = np.array(tp_emb).reshape(-1, 1)
        preds = reranker.predict_bleu_score(text_seqs, da_seqs, tp_seqs)
        for pred, text in zip(preds, texts):
            bleu.reset()
            bleu.append(text, true_texts)
            real = bleu.score()
            mapping.append((pred[0], real))

    print(mapping)
    preds = [x for x, _ in mapping]
    reals = [x for _, x in mapping]
    plt.scatter(reals, preds, alpha=0.1)
    plt.plot([0, 1], [0, 1], color='red')
    plt.show()
