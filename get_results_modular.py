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
models = TGEN_Model(da_embedder, text_embedder, cfg['tgen_seq2seq_config'])
models.load_models()


should_use_cache = "populate_greedy_cache" in cfg and cfg["populate_greedy_cache"]
should_update_cache = should_use_cache and "update_greedy_cache" in cfg and cfg["update_greedy_cache"]
if should_use_cache:
    models.populate_cache()

absts = get_abstss_test()
for beam_size in cfg["beam_sizes"]:
    print("Beam size = {} ".format(beam_size))
    beam_save_path = cfg.get('beam_save_path', '')
    if beam_save_path:
        beam_save_path = beam_save_path.format(beam_size)
    scorer_func = get_score_function(cfg['scorer'], cfg, models, true_vals, beam_size)
    preds = run_beam_search_with_rescorer(scorer_func, models, das_test, beam_size, cfg['only_rerank_final'],
                                          beam_save_path, should_save_cache=should_update_cache)
    preds = [[x for x in pred if x not in [START_TOK, END_TOK, PAD_TOK]] for pred in preds]
    if "res_save_format" in cfg:
        save_filename = cfg["res_save_format"].format(beam_size)
    elif cfg['scorer'] in ['surrogate', 'greedy_decode_surrogate']:
        # Example surrogate-regression_reranker_relative-categorical_order_10_10.txt
        surrogate_cfg = yaml.load(open(cfg["trainable_reranker_config"],'r+'))
        save_filename = "{}-{}-{}-{}-{}.txt".format(cfg['scorer'],surrogate_cfg["output_type"], surrogate_cfg["logprob_preprocess_type"], surrogate_cfg['beam_size'], beam_size)
    else:
        exit(0)
    save_path = os.path.join(RESULTS_DIR, save_filename)
    post_abstr = apply_absts(absts, preds)
    print("Saving to {}".format(save_path))
    with open(save_path, "w+") as out_file:
        for pa in post_abstr:
            out_file.write(" ".join(pa) + '\n')


