import random
import numpy as np
from keras.utils import to_categorical

from base_models import TGEN_Reranker, TrainableReranker
from e2e_metrics.metrics.pymteval import BLEUScore
from utils import START_TOK, END_TOK, PAD_TOK, get_features


def get_regressor_score_func(regressor, text_embedder, w2v):
    def func(path, logprob, da_emb, da_i, enc_outs):
        features = get_features(path, text_embedder, w2v, logprob)
        regressor_score = regressor.predict(features.reshape(1, -1))[0][0]
        return regressor_score

    return func


def get_tgen_rerank_score_func(tgen_reranker, da_embedder):
    def func(path, logprob, da_emb, da_i, enc_outs):
        text_emb = path[1]
        reranker_score = tgen_reranker.get_pred_hamming_dist(text_emb, da_emb, da_embedder)
        return path[0] - 100 * reranker_score

    return func


def get_identity_score_func():
    def func(path, logprob, da_emb, da_i, enc_outs):
        return path[0]

    return func


def get_greedy_decode_score_func(models, final_scorer, max_length_out, save_scores=None):
    def func(path, logprob, da_emb, da_i, enc_outs):
        path = models.naive_complete_greedy(path, enc_outs, max_length_out - len(path[1]))
        score = final_scorer(path, logprob, da_emb, da_i, enc_outs)
        if save_scores is not None:
            if type(save_scores) is dict:
                raise NotImplementedError("This bit need to be rewritten")
                # text_emb = models.text_embedder.get_embeddings(tokenised_texts=[toks])[0]
                # text_emb = models.text_embedder.remove_pad_from_embed(text_emb)
                # da_emb = models.da_embedder.remove_pad_from_embed(da_emb)
                # save_scores[(tuple(da_emb), tuple(text_emb))] = (score, log_prob)
        return score

    return func


def get_oracle_score_func(bleu, true_vals, text_embedder, reverse):
    def func(path, logprob, da_emb, da_i, enc_outs):
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
    def func(path, logprob, da_emb, da_i, enc_outs):
        return random.random()

    return func


def get_learned_score_func(trainable_reranker, test_beam_size, select_max=False, output_type=None):
    def func(path, logprob, da_emb, da_i, enc_outs):
        text_emb = path[1]
        pads = [trainable_reranker.text_embedder.tok_to_embed[PAD_TOK]] * \
               (trainable_reranker.text_embedder.length - len(text_emb))
        if trainable_reranker.logprob_preprocess_type == 'categorical_order':
            logprob_rank = logprob*trainable_reranker.beam_size // test_beam_size
            logprob_val = to_categorical([logprob_rank], num_classes=trainable_reranker.beam_size)
        else:
            logprob_val = [path[0]]

        pred = trainable_reranker.predict_bleu_score(
            np.array([pads + text_emb]),
            np.array([da_emb]),
            np.array(logprob_val))

        if trainable_reranker.output_type in ["regression_ranker", "regression_reranker_relative"]:
            return 1-pred[0][0]

        if select_max:
            max_pred = np.argmax(pred[0])
            return 10-max_pred, pred[0][0]
        else:
            return pred[0][0]

    return func


def get_score_function(scorer, cfg, models, true_vals, beam_size):
    da_embedder = models.da_embedder
    text_embedder = models.text_embedder
    print("Using Scorer: {}".format(scorer))
    if scorer == "TGEN":
        tgen_reranker = TGEN_Reranker(da_embedder, text_embedder, cfg['tgen_reranker_config'])
        tgen_reranker.load_model()
        return get_tgen_rerank_score_func(tgen_reranker, da_embedder)
    elif scorer == 'identity':
        return get_identity_score_func()
    elif scorer == 'greedy_decode_oracle':
        bleu_scorer = BLEUScore()
        final_scorer = get_oracle_score_func(bleu_scorer, true_vals, text_embedder, reverse=False)
        return get_greedy_decode_score_func(models, final_scorer=final_scorer, max_length_out=text_embedder.length)
    elif scorer == 'greedy_decode_tgen':
        tgen_reranker = TGEN_Reranker(da_embedder, text_embedder, cfg['tgen_reranker_config'])
        tgen_reranker.load_model()
        final_scorer = get_tgen_rerank_score_func(tgen_reranker, da_embedder)
        return get_greedy_decode_score_func(models, final_scorer=final_scorer, max_length_out=text_embedder.length)
    elif scorer == 'greedy_decode_surrogate':
        learned = TrainableReranker(da_embedder, text_embedder, cfg['trainable_reranker_config'])
        learned.load_model()
        final_scorer = get_learned_score_func(learned, beam_size, output_type=cfg["output_type"])
        return get_greedy_decode_score_func(models, final_scorer=final_scorer, max_length_out=text_embedder.length)
    elif scorer == 'greedy_id':
        final_scorer = get_identity_score_func()
        return get_greedy_decode_score_func(models, final_scorer=final_scorer, max_length_out=text_embedder.length)
    elif scorer in ['oracle', 'rev_oracle']:
        bleu_scorer = BLEUScore()
        return get_oracle_score_func(bleu_scorer, true_vals, text_embedder, reverse=(scorer == 'rev_oracle'))
    elif scorer == 'surrogate':
        learned = TrainableReranker(da_embedder, text_embedder, cfg['trainable_reranker_config'])
        learned.load_model()
        print(cfg)
        select_max = cfg.get("order_by_max_class", False)
        return get_learned_score_func(learned, beam_size, select_max)
    elif scorer == 'random':
        return get_random_score_func()
    else:
        raise ValueError("Unknown Scorer {}".format(cfg['scorer']))
