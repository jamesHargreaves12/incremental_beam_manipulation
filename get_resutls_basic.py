import os
import sys
import yaml

from base_model import TGEN_Model, TGEN_Reranker
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from reimplement_reinforce import run_beam_search_with_rescorer, get_tgen_rerank_score_func, get_identity_score_func
from utils import get_training_variables, apply_absts, get_abstss, get_test_das, START_TOK, END_TOK, PAD_TOK

cfg = yaml.load(open("configs/run_vanilla_results.yaml", "r"))
texts, das, = get_training_variables()
text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
da_embedder = DAEmbeddingSeq2SeqExtractor(das)

das_test = get_test_das()

if cfg['scorer'] == "TGEN":
    tgen_reranker = TGEN_Reranker(da_embedder, text_embedder, cfg)
    tgen_reranker.load_models_from_location(cfg['reranker_loc'])
    scorer_func = get_tgen_rerank_score_func(tgen_reranker, da_embedder)
elif cfg['scorer'] == 'identity':
    scorer_func = get_identity_score_func()
else:
    raise ValueError("Unknown Scorer {}".format(cfg['scorer']))

models = TGEN_Model(da_embedder, text_embedder, cfg)
absts = get_abstss()
for beam_size in [3]:
    preds = run_beam_search_with_rescorer(scorer_func, models, None, None, text_embedder,
                                          da_embedder, das_test, beam_size)
    preds = [[x for x in pred if x not in [START_TOK, END_TOK, PAD_TOK]] for pred in preds]
    post_abstr = apply_absts(absts, preds)
    with open(cfg["res_save_format"].format(beam_size), "w+") as out_file:
        for pa in post_abstr:
            out_file.write(" ".join(pa) + '\n')
