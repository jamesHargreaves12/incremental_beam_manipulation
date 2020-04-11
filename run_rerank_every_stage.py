# import os
# import sys
# import yaml
#
# from base_model import TGEN_Model, TGEN_Reranker
# from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
# from reimplement_reinforce import run_beam_search_with_rescorer, get_tgen_rerank_score_func, get_identity_score_func
# from utils import get_texts_training
# sys.path.append(os.path.join(os.getcwd(), 'tgen'))
#
# from tgen.futil import read_das
#
# cfg = yaml.load(open("configs/config_train.yaml", "r"))
# use_size = cfg['use_size']
# valid_size = cfg['valid_size']
# epoch = cfg['epoch']
# batch_size = cfg['batch_size']
# hidden_size = cfg['hidden_size']
# embedding_size = cfg['embedding_size']
# load_from_save = cfg['load_from_save']
# train_reranker = True
# reranker_loc = "models/reranker/tgen"
# das = read_das("tgen/e2e-challenge/input/train-das.txt")
#
# texts = [['<S>'] + x + ['<E>'] for x in get_texts_training()]
# print(texts[0])
#
# text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
# da_embedder = DAEmbeddingSeq2SeqExtractor(das)
#
# das = das[:use_size + valid_size]
# texts = texts[:use_size + valid_size]
#
# text_embs = text_embedder.get_embeddings(texts)
# text_vsize = text_embedder.vocab_length
# text_len = len(text_embs[0])
#
# da_embs = da_embedder.get_embeddings(das)
# da_vsize = da_embedder.vocab_length
# da_len = len(da_embs[0])
#
# scorer = 'identity'
# if scorer == "TGEN":
#     tgen_reranker = TGEN_Reranker(text_len, text_vsize, da_embedder.inclusion_length, text_embedder, cfg)
#     tgen_reranker.load_models_from_location(reranker_loc)
#     scorer_func = get_tgen_rerank_score_func(tgen_reranker, da_embedder)
# elif scorer == 'identity':
#     scorer_func = get_identity_score_func()
#
# for beam_size in [3]:
#     models = TGEN_Model(da_len, text_len, da_vsize, text_vsize, beam_size, cfg)
#     for text, da in zip(texts, das):
#         result = run_beam_search_with_rescorer(scorer_func, models, [da], beam_size)
#         print(result)
#
