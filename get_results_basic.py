import os
import sys
import yaml

from base_model import TGEN_Model, TGEN_Reranker
from e2e_metrics.metrics.pymteval import BLEUScore
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from reimplement_reinforce import run_beam_search_with_rescorer, get_tgen_rerank_score_func, get_identity_score_func, \
    get_greedy_decode_score_func
from utils import get_training_variables, apply_absts, get_abstss, get_test_das, START_TOK, END_TOK, PAD_TOK, \
    get_true_sents

cfg_path = "configs/run_vanilla_results.yaml"
print("Using config from: {}".format(cfg_path))
cfg = yaml.load(open(cfg_path, "r"))
texts, das, = get_training_variables()
text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
da_embedder = DAEmbeddingSeq2SeqExtractor(das)

das_test = get_test_das()
true_vals = get_true_sents()
models = TGEN_Model(da_embedder, text_embedder, cfg)
models.load_models_from_location(cfg['model_save_loc'])

print("Using Scorer: {}".format(cfg["scorer"]))
if cfg['scorer'] == "TGEN":
    tgen_reranker = TGEN_Reranker(da_embedder, text_embedder, cfg)
    tgen_reranker.load_models_from_location(cfg['reranker_loc'])
    scorer_func = get_tgen_rerank_score_func(tgen_reranker, da_embedder)
elif cfg['scorer'] == 'identity':
    scorer_func = get_identity_score_func()
elif cfg['scorer'] == 'greedy_decode':
    bleu_scorer = BLEUScore()
    scorer_func = get_greedy_decode_score_func(models, bleu=bleu_scorer, true_vals=true_vals)
else:
    raise ValueError("Unknown Scorer {}".format(cfg['scorer']))


absts = get_abstss()
for beam_size in cfg["beam_sizes"]:
    print("Beam size = {} ".format(beam_size))
    preds = run_beam_search_with_rescorer(scorer_func, models, das_test, beam_size, cfg['only_rerank_final'])
    preds = [[x for x in pred if x not in [START_TOK, END_TOK, PAD_TOK]] for pred in preds]
    post_abstr = apply_absts(absts, preds)
    save_file = cfg["res_save_format"].format(beam_size)
    print("Saving to {}".format(save_file))
    with open(save_file, "w+") as out_file:
        for pa in post_abstr:
            out_file.write(" ".join(pa) + '\n')
