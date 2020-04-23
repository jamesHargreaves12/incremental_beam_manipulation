import argparse
import os
import sys
import yaml

from base_models import TGEN_Model, TGEN_Reranker
from e2e_metrics.metrics.pymteval import BLEUScore
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from reimplement_reinforce import run_beam_search_with_rescorer
from scorer_functions import get_score_function
from utils import get_training_variables, apply_absts, get_abstss_train, get_test_das, START_TOK, END_TOK, PAD_TOK, \
    get_true_sents, get_abstss_test, get_training_das_texts, RESULTS_DIR

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()

cfg_path = args.config_path
print("Using config from: {}".format(cfg_path))
cfg = yaml.load(open(cfg_path, "r"))
texts, das = get_training_variables()
text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
da_embedder = DAEmbeddingSeq2SeqExtractor(das)

das_test = get_test_das()
if "get_train_beam" in cfg and cfg["get_train_beam"]:
    das_test, _ = get_training_das_texts()

true_vals = get_true_sents()
models = TGEN_Model(da_embedder, text_embedder, cfg)
models.load_models()

scorer_func = get_score_function(cfg['scorer'], cfg, models, true_vals)

should_use_cache = "populate_greedy_cache" in cfg and cfg["populate_greedy_cache"]
if should_use_cache:
    models.populate_cache()

absts = get_abstss_test()
for beam_size in cfg["beam_sizes"]:
    print("Beam size = {} ".format(beam_size))
    preds = run_beam_search_with_rescorer(scorer_func, models, das_test, beam_size, cfg['only_rerank_final'],
                                          cfg.get('beam_save_path', None), should_save_cache=should_use_cache)
    preds = [[x for x in pred if x not in [START_TOK, END_TOK, PAD_TOK]] for pred in preds]
    if "res_save_format" in cfg:
        save_filename = cfg["res_save_format"].format(beam_size)
        save_path = os.path.join(RESULTS_DIR, save_filename)
        post_abstr = apply_absts(absts, preds)
        print("Saving to {}".format(save_path))
        with open(save_path, "w+") as out_file:
            for pa in post_abstr:
                out_file.write(" ".join(pa) + '\n')
