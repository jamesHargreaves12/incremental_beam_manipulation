import argparse
import os
import sys
import yaml
from tqdm import tqdm

from base_models import TGEN_Model, TGEN_Reranker
from e2e_metrics.metrics.pymteval import BLEUScore
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from reimplement_reinforce import run_beam_search_with_rescorer
from scorer_functions import get_score_function
from utils import get_training_variables, apply_absts, get_abstss_train, get_test_das, START_TOK, END_TOK, PAD_TOK, \
    get_true_sents, get_final_beam, get_abstss_test, RESULTS_DIR

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()


cfg_path = args.config_path
print("Using config from: {}".format(cfg_path))
cfg = yaml.load(open(cfg_path, "r"))
texts, das, = get_training_variables()
text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
da_embedder = DAEmbeddingSeq2SeqExtractor(das)

das_test = get_test_das()
true_vals = get_true_sents()
models = TGEN_Model(da_embedder, text_embedder, cfg["tgen_seq2seq_config"])
models.load_models()

scorer_func = get_score_function(cfg['scorer'], cfg, models, true_vals)

absts = get_abstss_test()

for beam_size in cfg["beam_sizes"]:
    print("Beam size = {} ".format(beam_size))
    final_beams = get_final_beam(beam_size)
    preds = []
    for i, (beam, da_emb) in tqdm(list(enumerate(zip(final_beams, da_embedder.get_embeddings(das_test))))):
        scores = []
        enc_outs = None #only needed for greedy decode
        for hyp, logprob in beam:
            fake_path = (logprob, text_embedder.get_embeddings([hyp], pad_from_end=False)[0])
            # NOTE THAT THE TOKEN PROB IS SET TO 0 SINCE WE DONT HAVE IT
            tp = 0
            score = scorer_func(fake_path, tp, da_emb, i, enc_outs)
            scores.append((score, hyp))
        pred = sorted(scores, reverse=True)[0][1]
        preds.append(pred)

    preds = [[x for x in pred if x not in [START_TOK, END_TOK, PAD_TOK]] for pred in preds]

    post_abstr = apply_absts(absts, preds)
    save_filename = cfg["res_save_format"].format(beam_size)
    save_path = os.path.join(RESULTS_DIR, save_filename)
    print("Saving to {}".format(save_path))
    with open(save_path, "w+") as out_file:
        for pa in post_abstr:
            out_file.write(" ".join(pa) + '\n')
