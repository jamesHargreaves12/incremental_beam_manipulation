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
from tqdm import tqdm

from base_model import TGEN_Model, Regressor
from e2e_metrics.metrics.pymteval import BLEUScore
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from utils import get_texts_training, RERANK, get_training_das_texts, safe_get_w2v, apply_absts, PAD_TOK, END_TOK, \
    START_TOK


def get_features(path, text_embedder, w2v, tok_prob):
    h = path[2][0][0]
    c = path[2][1][0]
    pred_words = [text_embedder.embed_to_tok[x] for x in path[1]]

    return np.concatenate((h, c,
                           safe_get_w2v(w2v, pred_words[-1]), safe_get_w2v(w2v, pred_words[-2]),
                           [tok_prob, path[0], len(pred_words)]))


def load_rein_data(filepath):
    with open(filepath, "r") as fp:
        features = []
        labs = []
        for line in fp.readlines():
            line = [float(x) for x in line.split(",")]
            labs.append(line[-1])
            features.append(line[:-1])
        return features, labs


def get_completion_score(beam_search_model, da_emb, path, bleu, true):
    text_embedder = beam_search_model.text_embedder
    cur = " ".join(text_embedder.reverse_embedding(path[1])[:-1])
    rest = beam_search_model.make_prediction(da_emb,
                                             beam_size=1,
                                             prev_tok=text_embedder.embed_to_tok[path[1][-1]],
                                             max_length=2 * len(true) - len(path[1]))
    bleu.reset()
    bleu.append(cur + " " + rest, true)
    return bleu.score()


def reinforce_learning(beam_size, data_save_path, beam_search_model: TGEN_Model, das, truth, regressor, text_embedder,
                       da_embedder, cfg,
                       chance_of_choosing=0.01):
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
                        ref_score = get_completion_score(beam_search_model, da_emb, path, bleu, [true])
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
        test_res = run_beam_search_with_rescorer(regressor_scorer, beam_search_model, das[:1])
        print(" ".join(test_res[0]))


def get_regressor_score_func(regressor, text_embedder, w2v):
    def func(path, tp, da_emb):
        features = get_features(path, text_embedder, w2v, tp)
        regressor_score = regressor.predict(features.reshape(1, -1))[0][0]
        return regressor_score

    return func


def get_tgen_rerank_score_func(tgen_reranker, da_embedder):
    def func(path, tp, da_emb, da_i):
        text_emb = path[1]
        reranker_score = tgen_reranker.get_pred_hamming_dist(text_emb, da_emb, da_embedder)
        return path[0] - 100 * reranker_score

    return func


def get_identity_score_func():
    def func(path, tp, da_emb, da_i):
        return path[0]

    return func


def get_greedy_decode_score_func(models, bleu, true_vals):
    def func(path, tp, da_emb, da_i):
        true = true_vals[da_i]
        return get_completion_score(models, da_emb, path, bleu, true)

    return func


def get_oracle_score_func(bleu, true_vals, text_embedder, reverse):
    def func(path, tp, da_emb, da_i):
        true = true_vals[da_i]
        toks = text_embedder.reverse_embedding(path[1])
        pred = [x for x in toks if x not in [START_TOK, END_TOK, PAD_TOK]]
        bleu.reset()
        bleu.append(pred, true)
        if reverse:
            return 1 - bleu.score()
        return bleu.score()

    return func


def get_random_score_func():
    def func(path, tp, da_emb, da_i):
        return random.random()

    return func


def run_beam_search_with_rescorer(scorer, beam_search_model: TGEN_Model, das, beam_size, only_rescore_final=False,
                                  save_final_beam_path=False):
    da_embedder = beam_search_model.da_embedder
    text_embedder = beam_search_model.text_embedder
    max_predict_len = 60

    results = []
    save_file = None
    if save_final_beam_path:
        save_file = open(save_final_beam_path.format(beam_size), "w+")
    for i, da_emb in tqdm(list(enumerate(da_embedder.get_embeddings(das)))):
        inf_enc_out = beam_search_model.encoder_model.predict(np.array([da_emb]))
        enc_outs = inf_enc_out[0]
        enc_last_state = inf_enc_out[1:]
        paths = [(log(1.0), text_embedder.start_emb, enc_last_state)]
        end_tokens = text_embedder.end_embs

        for step in range(max_predict_len):
            # expand
            new_paths, tok_probs = beam_search_model.beam_search_exapand(paths, enc_outs, beam_size)

            # prune
            path_scores = []
            for path, tp in zip(new_paths, tok_probs):
                if not only_rescore_final:
                    hyp_score = scorer(path, tp, da_emb, i)
                    path_scores.append((hyp_score, path))
                else:
                    path_scores.append((path[0], path))
            paths = [x[1] for x in sorted(path_scores, key=lambda y: y[0], reverse=True)[:beam_size]]

            if all([p[1][-1] in end_tokens for p in paths]):
                break

        if save_file:
            for path in paths:
                save_file.write(" ".join(text_embedder.reverse_embedding(path[1])) + " " + str(path[0]) + "\n")
            save_file.write("\n")

        if only_rescore_final:
            path_scores = []
            for path in paths:
                hyp_score = scorer(path, 1, da_emb, i)
                path_scores.append((hyp_score, path))
            paths = [x[1] for x in sorted(path_scores, key=lambda y: y[0], reverse=True)]

        best_path = paths[0]
        pred_toks = text_embedder.reverse_embedding(best_path[1])
        results.append(pred_toks)

    return results


if __name__ == "__main__":
    beam_size = 3
    cfg = yaml.load(open("config_reinforce.yaml", "r"))
    train_data_location = "output_files/training_data/{}_.csv".format(beam_size)
    das, texts = get_training_das_texts()
    print(das[0], texts[0])
    # sys.exit(0)
    text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
    da_embedder = DAEmbeddingSeq2SeqExtractor(das)
    das = das[:cfg['use_size']]
    texts = texts[:cfg['use_size']]
    n_in = cfg["hidden_size"] * 2 + cfg["w2v_size"] * 2 + 3

    text_vsize, text_len = text_embedder.vocab_length, text_embedder.length
    da_vsize, da_len = da_embedder.vocab_length, da_embedder.length
    print(da_vsize, text_vsize, da_len, text_len)

    models = TGEN_Model(da_embedder, text_embedder, cfg)
    models.load_models()

    regressor = Regressor(n_in, batch_size=1, max_len=max([len(x) for x in texts]))
    if cfg["classif_from_file"]:
        regressor.load_model(cfg["model_save_loc"])
    elif os.path.exists(train_data_location) and cfg["pretrain"]:
        feats, labs = load_rein_data(train_data_location)
        if feats:
            regressor.train(feats, labs)

    reinforce_learning(beam_size, train_data_location, models, das, texts, regressor, text_embedder, da_embedder, cfg)
    # save_path = "output_files/out-text-dir-v2/rein_{}.txt".format(beam_size)
    # absts = smart_load_absts('tgen/e2e-challenge/input/train-abst.txt')
    # print(run_classifier_bs(classifier, models, None, None, text_embedder, da_embedder, das[:1], beam_size, cfg))
