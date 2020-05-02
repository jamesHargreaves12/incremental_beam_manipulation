import pickle
from collections import defaultdict
from itertools import product
from math import log

from gensim.models import Word2Vec
import random
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from time import time
import numpy as np
import yaml
import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from tqdm import tqdm

from base_models import TGEN_Model, Regressor
from e2e_metrics.metrics.pymteval import BLEUScore
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from scorer_functions import get_identity_score_func
from utils import get_texts_training, RERANK, get_training_das_texts, safe_get_w2v, apply_absts, PAD_TOK, END_TOK, \
    START_TOK


# def load_rein_data(filepath):
#     with open(filepath, "r") as fp:
#         features = []
#         labs = []
#         for line in fp.readlines():
#             line = [float(x) for x in line.split(",")]
#             labs.append(line[-1])
#             features.append(line[:-1])
#         return features, labs


# def get_greedy_compelete_toks_logprob(beam_search_model, path, max_length, enc_outs):
#     text_embedder = beam_search_model.text_embedder
#
#     result = " ".join(text_embedder.reverse_embedding(path[1]))
#     if path[1][-1] not in text_embedder.end_embs:
#         rest, logprob = beam_search_model.complete_greedy(path, enc_outs, max_length)
#         if rest:
#             result = result + " " + rest
#
#     else:
#         logprob = path[0]
#     toks = [x for x in result.split(" ") if x not in [START_TOK, END_TOK, PAD_TOK]]
#     return toks, logprob


# def reinforce_learning(beam_size, data_save_path, beam_search_model: TGEN_Model, das, truth, regressor, text_embedder,
#                        da_embedder, cfg,
#                        chance_of_choosing=0.01):
#     # This needs to be updated
#     w2v = Word2Vec.load(cfg["w2v_path"])
#     D = []
#     bleu = BLEUScore()
#     bleu_overall = BLEUScore()
#     beam_search_proportion = 1.0
#     bsp_multiplier = cfg['reference_policy_decay_rate']
#
#     data_save_file = open(data_save_path, "a+")
#     for i in range(cfg["epoch"]):
#         for j, (da_emb, true) in tqdm(enumerate(zip(da_embedder.get_embeddings(das), truth))):
#             inf_enc_out = beam_search_model.encoder_model.predict(np.array([da_emb]))
#             enc_outs = inf_enc_out[0]
#             enc_last_state = inf_enc_out[1:]
#             paths = [(log(1.0), text_embedder.start_emb, enc_last_state)]
#             end_tokens = text_embedder.end_embs
#             final_scorer = final_scorer_bleu(bleu)
#             for step in range(len(true)):
#                 new_paths, tok_probs = beam_search_model.beam_search_exapand(paths, enc_outs, beam_size)
#
#                 path_scores = []
#                 regressor_order = random.random() > beam_search_proportion
#                 for path, tp in zip(new_paths, tok_probs):
#                     features = get_features(path, text_embedder, w2v, tp)
#                     if regressor_order:
#                         classif_score = regressor.predict(features.reshape(1, -1))[0][0]
#                         path_scores.append((classif_score, path))
#                     else:
#                         path_scores.append((path[0] + log(tp), path))
#
#                     # greedy decode
#                     if random.random() < chance_of_choosing:
#                         # this probs wont work as will have changed dramatically ********
#                         ref_score = get_completion_score(beam_search_model, path, final_scorer, [true], enc_outs)
#                         D.append((features, ref_score))
#                         data_save_file.write("{},{}\n".format(",".join([str(x) for x in features]), ref_score))
#                 # prune
#                 paths = [x[1] for x in sorted(path_scores, key=lambda y: y[0], reverse=True)[:beam_size]]
#
#                 if all([p[1][-1] in end_tokens for p in paths]):
#                     break
#             pred = text_embedder.reverse_embedding(paths[0][1])
#             bleu_overall.append(pred, [true])
#
#         score = bleu_overall.score()
#         bleu_overall.reset()
#         print("BLEU SCORE FOR last batch = {}".format(score))
#         features = [d[0] for d in D]
#         labs = [d[1] for d in D]
#         regressor.train(features, labs)
#         regressor.save_model(cfg["model_save_loc"])
#         beam_search_proportion *= bsp_multiplier
#         regressor_scorer = get_regressor_score_func(regressor, text_embedder, w2v)
#         test_res = run_beam_search_with_rescorer(regressor_scorer, beam_search_model, das[:1], )
#         print(" ".join(test_res[0]))
#

def score_beams_pairwise(beam, pair_wise_model, da_emb):
    text_embedder = pair_wise_model.text_embedder
    da_emb = np.array(da_emb)
    inf_beam_size = len(beam)
    lps = np.array([x[0] for x in beam])
    lps = pair_wise_model.setup_lps(lps)
    da_emb_set = []
    text_1_set = []
    text_2_set = []
    lp_1_set = []
    lp_2_set = []

    for i in range(inf_beam_size):
        text_1 = np.array([text_embedder.pad_to_length(beam[i][1])])
        lp_1 = lps[i]
        for j in range(i + 1, inf_beam_size):
            text_2 = np.array([text_embedder.pad_to_length(beam[j][1])])
            lp_2 = lps[j]

            da_emb_set.append(da_emb)
            text_1_set.append(text_1[0])
            text_2_set.append(text_2[0])
            lp_1_set.append(lp_1)
            lp_2_set.append(lp_2)
    da_emb_set, text_1_set, text_2_set, lp_1_set, lp_2_set = \
        np.array(da_emb_set), np.array(text_1_set), np.array(text_2_set), np.array(lp_1_set), np.array(lp_2_set)

    results = pair_wise_model.predict_order(da_emb_set, text_1_set, text_2_set, lp_1_set, lp_2_set)
    res_pos = 0
    tourn_wins = defaultdict(int)
    for i in range(inf_beam_size):
        for j in range(i + 1, inf_beam_size):
            if results[res_pos][0] > 0.5:
                tourn_wins[i] += 1
            else:
                tourn_wins[j] += 1
            res_pos += 1
    scores = [(tourn_wins[i], beam[i]) for i in range(inf_beam_size)]
    return scores


def score_beams(rescorer, beam, da_emb, i, enc_outs):
    path_scores = []
    logprobs = [x[0] for x in beam]
    for path in beam:
        lp_pos = sum([1 for lp in logprobs if lp > path[0] + 0.000001])
        hyp_score = rescorer(path, lp_pos, da_emb, i, enc_outs)
        path_scores.append((hyp_score, path))
    return path_scores


def order_beam_acording_to_rescorer(rescorer, beam, da_emb, i, enc_outs, pairwise_flag):
    # In a horrible way of doing things the pairwise model is passed as the rescorer
    if pairwise_flag:
        path_scores = score_beams_pairwise(beam, rescorer, da_emb)
    else:
        path_scores = score_beams(rescorer, beam, da_emb, i, enc_outs)
    sorted_paths = sorted(path_scores, reverse=True, key=lambda x: x[0])
    return [x[1] for x in sorted_paths]


def order_beam_after_greedy_complete(rescorer, beam, da_emb, i, enc_outs, seq2seq, max_pred_len):
    finished_beam = beam.copy()
    for step in range(max_pred_len):
        finished_beam, _ = seq2seq.beam_search_exapand(finished_beam, enc_outs, 1)
        if all([p[1][-1] in seq2seq.text_embedder.end_embs for p in finished_beam]):
            break
    scored_finished_beams = score_beams(rescorer, finished_beam, da_emb, i, enc_outs)
    order = sorted(enumerate(scored_finished_beams), reverse=True, key=lambda x: x[1][0])
    result = [beam[i] for i,_ in order]
    return result


def _run_beam_search_with_rescorer(i, da_emb, paths, enc_outs, beam_size, max_pred_len, seq2seq,
                                   rescorer=None, greedy_complete=[], pairwise_flag=False,
                                   save_progress_file=None):
    end_tokens = seq2seq.text_embedder.end_embs
    for step in range(max_pred_len):
        # expand
        new_paths, tok_probs = seq2seq.beam_search_exapand(paths, enc_outs, beam_size)
        # prune
        if step in greedy_complete:
            paths = order_beam_after_greedy_complete(rescorer, new_paths, da_emb, i, enc_outs, seq2seq, max_pred_len)
        else:
            paths = order_beam_acording_to_rescorer(rescorer, new_paths, da_emb, i, enc_outs, pairwise_flag)
        paths = paths[:beam_size]

        if save_progress_file:
            save_progress_file.write("Step: {}\n".format(step))
            for path in paths:
                toks = [x for x in seq2seq.text_embedder.reverse_embedding(path[1]) if x != PAD_TOK]
                save_progress_file.write(" ".join(toks) + '\n')
            save_progress_file.write("\n")
        if all([p[1][-1] in end_tokens for p in paths]):
            break
    return paths


def run_beam_search_with_rescorer(scorer, beam_search_model: TGEN_Model, das, beam_size, only_rerank_final=False,
                                  save_final_beam_path='', greedy_complete=[], pairwise_flag=False,
                                  max_pred_len=60, save_progress_path=None):
    if save_progress_path is not None:
        save_progress_file = open(save_progress_path.format(beam_size), 'w+')
    else:
        save_progress_file = None
    da_embedder = beam_search_model.da_embedder
    text_embedder = beam_search_model.text_embedder

    results = []
    should_save_beams = save_final_beam_path and not os.path.exists(save_final_beam_path) and only_rerank_final
    should_load_beams = save_final_beam_path and os.path.exists(save_final_beam_path) and only_rerank_final
    load_final_beams = []
    final_beams = []
    if should_load_beams:
        load_final_beams = pickle.load((open(save_final_beam_path, "rb")))

    for i, da_emb in tqdm(list(enumerate(da_embedder.get_embeddings(das)))):
        if save_progress_file:
            save_progress_file.write("Test {}\n".format(i))
        inf_enc_out = beam_search_model.encoder_model.predict(np.array([da_emb]))
        enc_outs = inf_enc_out[0]
        enc_last_state = inf_enc_out[1:]
        paths = [(log(1.0), text_embedder.start_emb, enc_last_state)]

        if should_load_beams:
            paths = load_final_beams[i]
        else:
            paths = _run_beam_search_with_rescorer(
                i=i,
                da_emb=da_emb,
                paths=paths,
                enc_outs=enc_outs,
                beam_size=beam_size,
                max_pred_len=max_pred_len,
                seq2seq=beam_search_model,
                rescorer=scorer if not only_rerank_final else get_identity_score_func(),
                greedy_complete=greedy_complete,
                pairwise_flag=pairwise_flag,
                save_progress_file=save_progress_file
            )

        final_beams.append(paths)

        if only_rerank_final:
            paths = order_beam_acording_to_rescorer(scorer, paths, da_emb, i, enc_outs, pairwise_flag)
            if i == 0:
                print("First beam Score Distribution:")
                print([x[0] for x in paths])
                print("******************************")

        best_path = paths[0]
        pred_toks = text_embedder.reverse_embedding(best_path[1])
        results.append(pred_toks)

    if should_save_beams:
        print("Saving final beam states at ", save_final_beam_path)
        pickle.dump(final_beams, open(save_final_beam_path, "wb+"))

    return results


def get_best_from_beam_pairwise(beam, pair_wise_model, da_emb):
    path_scores = score_beams_pairwise(beam, pair_wise_model, da_emb)
    sorted_paths = sorted(path_scores, reverse=True, key=lambda x: x[0])
    return [x[1] for x in sorted_paths]


# def run_beam_search_pairwise(beam_search_model: TGEN_Model, das, beam_size, pairwise_model, only_rerank_final=False,
#                              save_final_beam_path=''):
#     results = []
#     load_final_beams = pickle.load((open(save_final_beam_path, "rb")))
#     from_saved_beams = only_rerank_final
#     max_pred_length = 60
#
#     for i, da_emb in tqdm(list(enumerate(beam_search_model.da_embedder.get_embeddings(das)))):
#         if from_saved_beams:
#             paths = load_final_beams[i]
#         else:
#             inf_enc_out = beam_search_model.encoder_model.predict(np.array([da_emb]))
#             enc_outs = inf_enc_out[0]
#             enc_last_state = inf_enc_out[1:]
#             init_path = [(log(1.0), beam_search_model.text_embedder.start_emb, enc_last_state)]
#
#             paths = _run_beam_search_with_rescorer(
#                 i=i,
#                 da_emb=da_emb,
#                 paths=init_path,
#                 enc_outs=enc_outs,
#                 beam_size=beam_size,
#                 max_pred_len=max_pred_length,
#                 beam_search_model=beam_search_model,
#                 rescorer=scorer if not only_rerank_final else None
#             )
#
#         best_path = get_best_from_beam_pairwise(paths, pairwise_model, da_emb, beam_search_model.text_embedder)
#         pred_toks = beam_search_model.text_embedder.reverse_embedding(best_path[1])
#         results.append(pred_toks)
#
#     return results
