# # ./run_tgen.py seq2seq_train
# import os
# import sys
#
# from tqdm import tqdm
#
# from e2e_metrics.metrics.pymteval import BLEUScore
#
# sys.path.append(os.path.join(os.getcwd(), 'tgen'))
# from utils import construct_logs, RERANK, get_texts_training
# from getopt import getopt
#
# from old_files.beam_search_edit import _beam_search, lexicalize_beam, save_training_data, rolling_beam_search
# from e2e_metrics.measure_scores import load_data, evaluate, run_pymteval
#
# import nltk
# import numpy as np
# from nltk.translate.bleu_score import SmoothingFunction
# from tgen.tree import TreeData
# from tgen.tfclassif import Reranker
# from tgen.config import Config
# from tgen.seq2seq import Seq2SeqGen, Seq2SeqBase, cut_batch_into_steps
# from tgen.logf import log_info, log_debug, log_warn
# from tgen.futil import read_das, write_ttrees, write_tokens, postprocess_tokens, create_ttree_doc, smart_load_absts
#
# import tensorflow as tf
#
# true_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/tgen/e2e-challenge/input/devel-text.txt"
#
# def process_das(beam_size, da, tgen, absts, reranker, true_sents=[], test_pruner=False):
#     # pass one at a time
#
#     enc_inputs = np.array([[x] for x in tgen.da_embs.get_embeddings(da)])
#     dec_output_ids = _beam_search(tgen, enc_inputs, da, beam_size, absts, reranker, true_sents, test_pruner)
#
#     dec_trees = [tgen.tree_embs.ids_to_tree(ids) for ids in dec_output_ids.transpose()]
#
#     return dec_trees
#
#
# def gen_training_data_for_da(tgen, da, da_id, true_sents, beam_size):
#     enc_inputs = np.array([[x] for x in tgen.da_embs.get_embeddings(da)])
#     rolling_beam_search(
#         tgen=tgen,
#         enc_inputs=enc_inputs,
#         beam_size=beam_size,
#         da_id=da_id,
#         true_sents=true_sents
#     )
#
# da_train_file = "tgen/e2e-challenge/input/train-das.txt"
# text_train_file = "tgen/e2e-challenge/input/train-text.txt"
# seq2seq_config_file = "tgen/e2e-challenge/config/config.yaml"
# da_test_file = "tgen/e2e-challenge/input/devel-das.txt"
# test_abstr_file = "tgen/e2e-challenge/input/devel-abst.txt"
# train_abstr_file = "tgen/e2e-challenge/input/train-abst.txt"
#
#
# def train(seq2seq_model_file="models/model_e2e_2/model.pickle.gz"):
#     tf.reset_default_graph()
#     parent_dir = os.path.abspath(os.path.join(seq2seq_model_file, os.pardir))
#     if not os.path.exists(parent_dir):
#         os.mkdir(parent_dir)
#
#     print('Training sequence-to-sequence generator...')
#     config = Config(seq2seq_config_file)
#
#     generator = Seq2SeqGen(config)
#
#     generator.train(da_train_file, text_train_file)
#     sys.setrecursionlimit(100000)
#     generator.save_to_file(seq2seq_model_file)
#
#
# def gen_training_data(beam_size, seq2seq_model_file="models/model_e2e_2/model.pickle.gz"):
#     tf.reset_default_graph()
#
#     tgen = Seq2SeqBase.load_from_file(seq2seq_model_file)
#     tgen.beam_size = beam_size
#
#     das = read_das(da_train_file)
#     truth = [[x] for x in get_texts_training()]
#     print('Generating...')
#     assert(len(das) == len(truth))
#     for i, (da,true) in tqdm(enumerate(zip(das,truth))):
#         gen_training_data_for_da(tgen, da, i, true, beam_size)
#     save_training_data("output_files/training_data/{}_v3_big.csv".format(beam_size))
#
#
# # TODO add so doesnt overwrite files e.g. v1, v2 ect
# def test_gen(beam_size,
#              seq2seq_model_file="models/model_e2e_2/model.pickle.gz",
#              reranker=RERANK.VANILLA, bc_classif=False):
#     tf.reset_default_graph()
#
#     tgen = Seq2SeqBase.load_from_file(seq2seq_model_file)
#     tgen.beam_size = beam_size
#
#     if reranker in [RERANK.ORACLE, RERANK.REVERSE_ORACLE]:
#         true_sentences = []
#         with open(true_file, "r") as true:
#             current_set = []
#             for line in true:
#                 if len(line) > 1:
#                     current_set.append(line.strip('\n').split(" "))
#                 else:
#                     true_sentences.append(current_set)
#                     current_set = []
#
#     output_file = "output_files/out-text-dir-v2/{}_b-{}.txt".format(reranker, beam_size)
#     parent_dir = os.path.abspath(os.path.join(output_file, os.pardir))
#     if not os.path.exists(parent_dir):
#         os.mkdir(parent_dir)
#
#     # read input files (DAs, contexts)
#     das = read_das(da_test_file)
#     # generate
#     print('Generating...')
#     # print("DAS:", das[0])
#
#     abstss = smart_load_absts(test_abstr_file, len(das))
#
#     if reranker not in [RERANK.ORACLE, RERANK.REVERSE_ORACLE]:
#         gen_trees = [process_das(beam_size, da, tgen, absts, reranker, test_pruner=bc_classif) for da, absts in tqdm(zip(das, abstss))]
#     else:
#         gen_trees = [process_das(beam_size, da, tgen, absts, reranker, ts, test_pruner=bc_classif) for da, absts, ts in
#                      tqdm(zip(das, abstss, true_sentences))]
#
#     print('Lexicalizing...')
#     for i in range(len(gen_trees)):
#         lexicalize_beam(tgen.lexicalizer, gen_trees[i], abstss[i])
#
#     print('Writing output...')
#     if output_file.endswith('.txt'):
#         gen_toks = [t[0].to_tok_list() for t in gen_trees]
#         postprocess_tokens(gen_toks, das)
#         write_tokens(gen_toks, output_file)
#
#
# def test_res_official(beam_size, reranker=RERANK.VANILLA):
#     pred_file = "output_files/out-text-dir-v2/{}_b-{}.txt".format(reranker,
#                                                                   beam_size)
#     true_file = "tgen/e2e-challenge/input/devel-conc.txt"
#     data_src, data_ref, data_sys = load_data(true_file, pred_file)
#     # mteval_scores = run_pymteval(data_ref, data_sys)
#
#     bleu = BLEUScore()
#     # print(data_ref[0])
#     # print(data_sys[0])
#     for sents_ref, sent_sys in zip(data_ref, data_sys):
#         bleu.append(sent_sys, sents_ref)
#
#     # return the computed scores
#     bleu = bleu.score()
#
#     return bleu
#
#
# if __name__ == "__main__":
#     # train("models/model_e2e_3/model.pickle.gz")
#     reranker = 'oracle'
#     generate_results = False
#     get_results = True
#     training_data = False
#
#     print(reranker, generate_results, get_results)
#     # approx generation times:
#     # 3:       6
#     # 5:      10
#     # 10:     21
#     # 30:   1h20
#     # 100:  6h30
#
#     if generate_results:
#         for beam_size in [5, 10, 30, 100]:
#             construct_logs(beam_size)
#             test_gen(beam_size, reranker=reranker, bc_classif=True)
#
#     if get_results:
#         scores = []
#         for beam_size in [3, 5, 10, 30, 100]:
#             result = test_res_official(beam_size, reranker=reranker)
#             scores.append((beam_size, result))
#             print(beam_size, result)
#         print("Scores", scores)
#
#     if training_data:
#         beam_size = 5
#         gen_training_data(3)
