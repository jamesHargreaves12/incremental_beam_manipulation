import pickle
from collections import defaultdict
from itertools import product
from math import log

from gensim.models import Word2Vec
import random
import sys
import os
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
from utils import get_texts_training, RERANK, get_training_das_texts, safe_get_w2v, apply_absts, PAD_TOK, END_TOK, \
    START_TOK


def load_rein_data(filepath):
    with open(filepath, "r") as fp:
        features = []
        labs = []
        for line in fp.readlines():
            line = [float(x) for x in line.split(",")]
            labs.append(line[-1])
            features.append(line[:-1])
        return features, labs


def get_greedy_compelete_toks_logprob(beam_search_model, path, max_length, enc_outs):
    text_embedder = beam_search_model.text_embedder

    result = " ".join(text_embedder.reverse_embedding(path[1]))
    if path[1][-1] not in text_embedder.end_embs:
        rest, logprob = beam_search_model.complete_greedy(path, enc_outs, max_length)
        if rest:
            result = result + " " + rest

    else:
        logprob = path[0]
    toks = [x for x in result.split(" ") if x not in [START_TOK, END_TOK, PAD_TOK]]
    return toks, logprob


def reinforce_learning(beam_size, data_save_path, beam_search_model: TGEN_Model, das, truth, regressor, text_embedder,
                       da_embedder, cfg,
                       chance_of_choosing=0.01):
    # This needs to be updated
    w2v = Word2Vec.load(cfg["w2v_path"])
    D = []
    bleu = BLEUScore()
    bleu_overall = BLEUScore()
    beam_search_proportion = 1.0
    bsp_multiplier = cfg['reference_policy_decay_rate']

    data_save_file = open(data_save_path, "a+")
    for i in range(cfg["epoch"]):
        for j, (da_emb, true) in tqdm(enumerate(zip(da_embedder.get_embeddings(das), truth))):
            inf_enc_out = beam_search_model.encoder_model.predict(np.array([da_emb]))
            enc_outs = inf_enc_out[0]
            enc_last_state = inf_enc_out[1:]
            paths = [(log(1.0), text_embedder.start_emb, enc_last_state)]
            end_tokens = text_embedder.end_embs
            final_scorer = final_scorer_bleu(bleu)
            for step in range(len(true)):
                new_paths, tok_probs = beam_search_model.beam_search_exapand(paths, enc_outs, beam_size)

                path_scores = []
                regressor_order = random.random() > beam_search_proportion
                for path, tp in zip(new_paths, tok_probs):
                    features = get_features(path, text_embedder, w2v, tp)
                    if regressor_order:
                        classif_score = regressor.predict(features.reshape(1, -1))[0][0]
                        path_scores.append((classif_score, path))
                    else:
                        path_scores.append((path[0] + log(tp), path))

                    # greedy decode
                    if random.random() < chance_of_choosing:
                        # this probs wont work as will have changed dramatically ********
                        ref_score = get_completion_score(beam_search_model, path, final_scorer, [true], enc_outs)
                        D.append((features, ref_score))
                        data_save_file.write("{},{}\n".format(",".join([str(x) for x in features]), ref_score))
                # prune
                paths = [x[1] for x in sorted(path_scores, key=lambda y: y[0], reverse=True)[:beam_size]]

                if all([p[1][-1] in end_tokens for p in paths]):
                    break
            pred = text_embedder.reverse_embedding(paths[0][1])
            bleu_overall.append(pred, [true])

        score = bleu_overall.score()
        bleu_overall.reset()
        print("BLEU SCORE FOR last batch = {}".format(score))
        features = [d[0] for d in D]
        labs = [d[1] for d in D]
        regressor.train(features, labs)
        regressor.save_model(cfg["model_save_loc"])
        beam_search_proportion *= bsp_multiplier
        regressor_scorer = get_regressor_score_func(regressor, text_embedder, w2v)
        test_res = run_beam_search_with_rescorer(regressor_scorer, beam_search_model, das[:1], )
        print(" ".join(test_res[0]))


def _run_beam_search_with_rescorer_indiv(i, da_emb, paths, enc_outs, beam_size, max_pred_len, beam_search_model,
                                         rescorer=None):
    end_tokens = beam_search_model.text_embedder.end_embs

    for step in range(max_pred_len):
        # expand
        new_paths, tok_probs = beam_search_model.beam_search_exapand(paths, enc_outs, beam_size)

        # prune
        path_scores = []
        logprobs = [x[0] for x in new_paths]
        for path, tp in zip(new_paths, tok_probs):
            if rescorer:
                lp_pos = sum([1 for lp in logprobs if lp > path[0] + 0.000001])
                lp_pos = lp_pos // beam_size  # the beam is 10x larger than would be expected by model
                hyp_score = rescorer(path, lp_pos, da_emb, i, enc_outs)
                path_scores.append((hyp_score, path))
            else:
                path_scores.append((path[0], path))
        paths = [x[1] for x in sorted(path_scores, key=lambda y: y[0], reverse=True)[:beam_size]]

        if all([p[1][-1] in end_tokens for p in paths]):
            break
    return paths


def run_beam_search_with_rescorer(scorer, beam_search_model: TGEN_Model, das, beam_size, only_rerank_final=False,
                                  save_final_beam_path='', should_save_cache=False, callback_1000=None):
    da_embedder = beam_search_model.da_embedder
    text_embedder = beam_search_model.text_embedder
    max_predict_len = 60

    results = []
    should_save_beams = save_final_beam_path and not os.path.exists(save_final_beam_path)
    should_load_beams = save_final_beam_path and os.path.exists(save_final_beam_path)
    load_final_beams = []
    save_final_beams = []
    if should_load_beams:
        load_final_beams = pickle.load((open(save_final_beam_path, "rb")))

    for i, da_emb in tqdm(list(enumerate(da_embedder.get_embeddings(das)))):
        inf_enc_out = beam_search_model.encoder_model.predict(np.array([da_emb]))
        enc_outs = inf_enc_out[0]
        enc_last_state = inf_enc_out[1:]
        paths = [(log(1.0), text_embedder.start_emb, enc_last_state)]

        if should_load_beams:
            paths = load_final_beams[i]
        else:
            paths = _run_beam_search_with_rescorer_indiv(
                i=i,
                da_emb=da_emb,
                paths=paths,
                enc_outs=enc_outs,
                beam_size=beam_size,
                max_pred_len=max_predict_len,
                beam_search_model=beam_search_model,
                rescorer=scorer if not only_rerank_final else None
            )

        if should_save_beams:
            save_final_beams.append(paths)

        if only_rerank_final:
            path_scores = []
            logprobs = [x[0] for x in paths]
            for path in paths:
                lp_pos = sum([1 for lp in logprobs if lp > path[0] + 0.000001])
                hyp_score = scorer(path, lp_pos, da_emb, i, enc_outs)
                path_scores.append((hyp_score, path))
            paths = [x[1] for x in sorted(path_scores, key=lambda y: y[0], reverse=True)]

            if i == 0:
                print("First beam Score Distribution:")
                print([x[0] for x in path_scores])
                print("******************************")

        best_path = paths[0]
        pred_toks = text_embedder.reverse_embedding(best_path[1])
        results.append(pred_toks)
        # print(" ".join(pred_toks))
        if i % 1000 == 0 and callback_1000 is not None:
            callback_1000(i)
    if should_save_beams:
        print("Saving final beam states at ", save_final_beam_path)
        pickle.dump(save_final_beams, open(save_final_beam_path, "wb+"))

    if should_save_cache:
        beam_search_model.save_cache()
        print("Cache Saved")
    return results


def get_best_from_beam_pairwise(beam, pair_wise_model, da_emb, text_embedder):
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
        for j in range(i+1, inf_beam_size):
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
        for j in range(i+1, inf_beam_size):
            if results[res_pos][0] > 0.5:
                tourn_wins[i] += 1
            else:
                tourn_wins[j] += 1
            res_pos += 1
    best = sorted(tourn_wins.items(), key=lambda x: x[1])[-1][0]
    return beam[best]



    # for _ in range(inf_beam_size):
    #     new_beam = []
    #     piv = random.randint(0, len(beam) - 1)
    #
    #     text_1 = np.array([text_embedder.pad_to_length(beam[piv][1])])
    #     lp_1 = lps[piv]
    #     for i in range(len(beam)):
    #         if i == piv:
    #             continue
    #         text_2 = np.array([text_embedder.pad_to_length(beam[i][1])])
    #         lp_2 = lps[i]
    #
    #         result = pair_wise_model.predict_order(da_emb, text_1, text_2, lp_1, lp_2)
    #         if result > 0.5:
    #             new_beam.append(beam[i])
    #     if not new_beam:
    #         return beam[piv]
    #     else:
    #         beam = new_beam
    # raise ValueError("Should never reach this point")


def run_beam_search_pairwise(beam_search_model: TGEN_Model, das, beam_size, pairwise_model, only_rerank_final=False,
                             save_final_beam_path=''):
    results = []
    load_final_beams = pickle.load((open(save_final_beam_path, "rb")))
    if not only_rerank_final:
        raise NotImplementedError("Currently only works for reranker")

    for i, da_emb in tqdm(list(enumerate(beam_search_model.da_embedder.get_embeddings(das)))):
        paths = load_final_beams[i]
        best_path = get_best_from_beam_pairwise(paths, pairwise_model, da_emb, beam_search_model.text_embedder)
        pred_toks = beam_search_model.text_embedder.reverse_embedding(best_path[1])
        results.append(pred_toks)

    return results
