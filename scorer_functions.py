from base_model import TGEN_Reranker
from e2e_metrics.metrics.pymteval import BLEUScore
from reimplement_reinforce import get_tgen_rerank_score_func, get_identity_score_func, get_greedy_decode_score_func, \
    get_oracle_score_func, get_random_score_func


def get_score_function(scorer, cfg, da_embedder, text_embedder, models, true_vals):
    print("Using Scorer: {}".format(scorer))
    if scorer == "TGEN":
        tgen_reranker = TGEN_Reranker(da_embedder, text_embedder, cfg)
        tgen_reranker.load_models_from_location(cfg['reranker_loc'])
        return get_tgen_rerank_score_func(tgen_reranker, da_embedder)
    elif scorer == 'identity':
        return get_identity_score_func()
    elif scorer == 'greedy_decode':
        bleu_scorer = BLEUScore()
        return get_greedy_decode_score_func(models, bleu=bleu_scorer, true_vals=true_vals)
    elif scorer in ['oracle', 'rev_oracle']:
        bleu_scorer = BLEUScore()
        return get_oracle_score_func(bleu_scorer, true_vals, text_embedder, scorer == 'rev_oracle')
    elif scorer == 'random':
        return get_random_score_func()
    else:
        raise ValueError("Unknown Scorer {}".format(cfg['scorer']))
