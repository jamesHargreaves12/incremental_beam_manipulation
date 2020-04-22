from base_model import TGEN_Reranker, TrainableReranker
from e2e_metrics.metrics.pymteval import BLEUScore
from reimplement_reinforce import get_tgen_rerank_score_func, get_identity_score_func, get_greedy_decode_score_func, \
    get_oracle_score_func, get_random_score_func, get_learned_score_func


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
