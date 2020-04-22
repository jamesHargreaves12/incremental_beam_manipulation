import random
import numpy as np
from base_models import TGEN_Reranker, TrainableReranker
from e2e_metrics.metrics.pymteval import BLEUScore
from reimplement_reinforce import get_greedy_compelete_toks_logprob
from utils import START_TOK, END_TOK, PAD_TOK, get_features


def get_regressor_score_func(regressor, text_embedder, w2v):
    def func(path, tp, da_emb, da_i, enc_outs):
        features = get_features(path, text_embedder, w2v, tp)
        regressor_score = regressor.predict(features.reshape(1, -1))[0][0]
        return regressor_score

    return func


def get_tgen_rerank_score_func(tgen_reranker, da_embedder):
    def func(path, tp, da_emb, da_i, enc_outs):
        text_emb = path[1]
        reranker_score = tgen_reranker.get_pred_hamming_dist(text_emb, da_emb, da_embedder)
        return path[0] - 100 * reranker_score

    return func


def get_identity_score_func():
    def func(path, tp, da_emb, da_i, enc_outs):
        return path[0]

    return func


def get_greedy_decode_score_func(models, final_scorer, max_length_out, save_scores=None):
    def func(path, tp, da_emb, da_i, enc_outs):
        toks, log_prob = get_greedy_compelete_toks_logprob(models, path, max_length_out - len(path[1]), enc_outs)
        text_emb = models.text_embedder.get_embeddings(tokenised_texts=[toks])[0]
        text_emb = models.text_embedder.remove_pad_from_embed(text_emb)
        comp_path = (log_prob, text_emb, path[2])
        # Working on the assumption that the final state of the decoder lstm is not required
        score = final_scorer(comp_path, tp, da_emb, da_i, enc_outs)
        if save_scores is not None:
            if type(save_scores) is dict:
                da_emb = models.da_embedder.remove_pad_from_embed(da_emb)
                save_scores[(tuple(da_emb), tuple(text_emb))] = score
        return score

    return func


def get_oracle_score_func(bleu, true_vals, text_embedder, reverse):
    def func(path, tp, da_emb, da_i, enc_outs):
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
    def func(path, tp, da_emb, da_i, enc_outs):
        return random.random()

    return func


def get_learned_score_func(trainable_reranker):
    def func(path, tp, da_emb, da_i, enc_outs):
        text_emb = path[1]
        pads = [trainable_reranker.text_embedder.tok_to_embed[PAD_TOK]] * \
               (trainable_reranker.text_embedder.length - len(text_emb))
        pred = trainable_reranker.predict_bleu_score(np.array([pads + text_emb]), np.array([da_emb]))
        return -pred[0][0]

    return func


def get_score_function(scorer, cfg, da_embedder, text_embedder, models, true_vals):
    print("Using Scorer: {}".format(scorer))
    if scorer == "TGEN":
        tgen_reranker = TGEN_Reranker(da_embedder, text_embedder, cfg)
        tgen_reranker.load_model()
        return get_tgen_rerank_score_func(tgen_reranker, da_embedder)
    elif scorer == 'identity':
        return get_identity_score_func()
    elif scorer == 'greedy_decode_oracle':
        bleu_scorer = BLEUScore()
        final_scorer = get_oracle_score_func(bleu_scorer, true_vals, text_embedder, reverse=False)
        return get_greedy_decode_score_func(models, final_scorer=final_scorer, max_length_out=text_embedder.length)
    elif scorer == 'greedy_decode_tgen':
        tgen_reranker = TGEN_Reranker(da_embedder, text_embedder, cfg)
        tgen_reranker.load_model()
        final_scorer = get_tgen_rerank_score_func(tgen_reranker, da_embedder)
        return get_greedy_decode_score_func(models, final_scorer=final_scorer, max_length_out=text_embedder.length)
    elif scorer in ['oracle', 'rev_oracle']:
        bleu_scorer = BLEUScore()
        return get_oracle_score_func(bleu_scorer, true_vals, text_embedder, reverse=(scorer == 'rev_oracle'))
    elif scorer == 'learned':
        learned = TrainableReranker(da_embedder, text_embedder, cfg)
        learned.load_model()
        return get_learned_score_func(learned)
    elif scorer == 'random':
        return get_random_score_func()
    else:
        raise ValueError("Unknown Scorer {}".format(cfg['scorer']))
