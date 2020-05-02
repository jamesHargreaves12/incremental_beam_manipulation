import argparse
import os
import sys
import yaml

from base_models import TGEN_Model, TGEN_Reranker, PairwiseReranker
from e2e_metrics.metrics.pymteval import BLEUScore
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from get_results_bleu_scores import print_results
from reimplement_reinforce import run_beam_search_with_rescorer
from scorer_functions import get_score_function
from utils import get_training_variables, apply_absts, get_abstss_train, get_test_das, START_TOK, END_TOK, PAD_TOK, \
    get_true_sents, get_abstss_test, get_training_das_texts, RESULTS_DIR, CONFIGS_MODEL_DIR, CONFIGS_DIR, postprocess

parser = argparse.ArgumentParser()
parser.add_argument('-c', default=None)
args = parser.parse_args()

cfg_path = args.c
if cfg_path is None:
    filenames = os.listdir(CONFIGS_DIR)
    filepaths = [os.path.join(CONFIGS_DIR, filename) for filename in filenames]
    mod_times = [(os.path.getmtime(x), i) for i, x in enumerate(filepaths) if not os.path.isdir(x)]
    cfg_path = filepaths[max(mod_times)[1]]

print("Using config from: {}".format(cfg_path))
cfg = yaml.safe_load(open(cfg_path, "r"))
texts, das = get_training_variables()
text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
da_embedder = DAEmbeddingSeq2SeqExtractor(das)

das_test = get_test_das()
if "get_train_beam" in cfg and cfg["get_train_beam"]:
    das_test, _ = get_training_das_texts()

true_vals = get_true_sents()
models = TGEN_Model(da_embedder, text_embedder, cfg['tgen_seq2seq_config'])
models.load_models()

absts = get_abstss_test()
for beam_size in cfg["beam_sizes"]:
    print("Beam size = {} ".format(beam_size))
    beam_save_path = cfg.get('beam_save_path', '')
    if beam_save_path:
        beam_save_path = beam_save_path.format(beam_size)
    # This is a horrible hack
    if cfg["pairwise_flag"]:
        scorer_func = PairwiseReranker(da_embedder, text_embedder, cfg["trainable_reranker_config"])
    else:
        scorer_func = get_score_function(cfg['scorer'], cfg, models, true_vals, beam_size)
    max_pred_len = 60
    if "greedy_complete_at" in cfg:
        greedy_complete = cfg["greedy_complete_at"]
    else:
        greedy_complete_rate = cfg.get("greedy_complete_rate", max_pred_len + 1)
        greedy_complete = list(range(greedy_complete_rate, max_pred_len, greedy_complete_rate))

    preds = run_beam_search_with_rescorer(scorer_func, models, das_test, beam_size,
                                          only_rerank_final=cfg['only_rerank_final'],
                                          save_final_beam_path=beam_save_path,
                                          greedy_complete=greedy_complete,
                                          pairwise_flag=cfg['pairwise_flag'],
                                          max_pred_len=60,
                                          save_progress_path=cfg.get('save_progress_file', None))

    preds = [[x for x in pred if x not in [START_TOK, END_TOK, PAD_TOK]] for pred in preds]
    if "res_save_format" in cfg:
        save_filename = cfg["res_save_format"].format(beam_size)
    elif cfg['scorer'] in ['surrogate', 'greedy_decode_surrogate']:
        # Example surrogate-regression_reranker_relative-categorical_order_10_10.txt
        surrogate_cfg = yaml.load(open(cfg["trainable_reranker_config"],'r+'))
        save_filename = "{}-{}-{}-{}-{}.txt".format(cfg['scorer'], surrogate_cfg["output_type"],
                                                    surrogate_cfg["logprob_preprocess_type"],
                                                    surrogate_cfg['beam_size'], beam_size)
    else:
        raise ValueError('Not saving files any where')
    save_path = os.path.join(RESULTS_DIR, save_filename)
    post_abstr = apply_absts(absts, preds)
    print("Saving to {}".format(save_path))
    with open(save_path, "w+") as out_file:
        for pa in post_abstr:
            out_file.write(postprocess(" ".join(pa)) + '\n')

print_results()


