import os
import sys
import yaml
from tqdm import tqdm

from base_model import TGEN_Model, TGEN_Reranker
from e2e_metrics.metrics.pymteval import BLEUScore
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from reimplement_reinforce import run_beam_search_with_rescorer, get_tgen_rerank_score_func, get_identity_score_func, \
    get_greedy_decode_score_func
from scorer_functions import get_score_function
from utils import get_training_variables, apply_absts, get_abstss, get_test_das, START_TOK, END_TOK, PAD_TOK, \
    get_true_sents, get_final_beam

cfg_path = "configs/run_rerank_random.yaml"
print("Using config from: {}".format(cfg_path))
cfg = yaml.load(open(cfg_path, "r"))
texts, das, = get_training_variables()
text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
da_embedder = DAEmbeddingSeq2SeqExtractor(das)

das_test = get_test_das()
true_vals = get_true_sents()
models = TGEN_Model(da_embedder, text_embedder, cfg)
models.load_models_from_location(cfg['model_save_loc'])

scorer_func = get_score_function(cfg['scorer'], cfg, da_embedder, text_embedder, models, true_vals)

absts = get_abstss()

for beam_size in cfg["beam_sizes"]:
    print("Beam size = {} ".format(beam_size))
    final_beams = get_final_beam(beam_size)
    preds = []
    for i, (beam, da_emb) in tqdm(list(enumerate(zip(final_beams, da_embedder.get_embeddings(das_test))))):
        scores = []
        for hyp, logprob in beam:
            fake_path = (logprob, text_embedder.get_embeddings([hyp], pad_from_end=False)[0])
            tp = 0
            score = scorer_func(fake_path, tp, da_emb, i)
            scores.append((score, hyp))
        pred = sorted(scores, reverse=True)[0][1]
        preds.append(pred)

    preds = [[x for x in pred if x not in [START_TOK, END_TOK, PAD_TOK]] for pred in preds]

    post_abstr = apply_absts(absts, preds)
    save_file = cfg["res_save_format"].format(beam_size)
    print("Saving to {}".format(save_file))
    with open(save_file, "w+") as out_file:
        for pa in post_abstr:
            out_file.write(" ".join(pa) + '\n')
