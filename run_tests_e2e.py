# ./run_tgen.py seq2seq_train
import os
import sys
from getopt import getopt


sys.path.append(os.path.join(os.getcwd(),'tgen'))

import nltk
from nltk.translate.bleu_score import SmoothingFunction
from tgen.tfclassif import Reranker
from tgen.config import Config
from tgen.seq2seq import Seq2SeqGen
import tensorflow as tf

from tgen.seq2seq import Seq2SeqBase

from  tgen.futil import read_das, write_ttrees, write_tokens, postprocess_tokens, create_ttree_doc

class RerankerOracle(Reranker):
    def __init__(self, cfg):
        Reranker.__init__(self, cfg)
        self.true_sentences = []
        true_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/tgen/e2e-challenge/input/devel-text.txt"
        with open(true_file, "r") as true:
            current_set = []
            for line in true:
                if len(line) > 1:
                    current_set.append(line.strip('\n').split(" "))
                else:
                    self.true_sentences.append(current_set)
                    current_set = []
    def dist_to_cur_da(self, trees):
        true_sents = self.true_sentences.pop()
        acc = []
        for tree in trees:
            acc.append(nltk.translate.bleu_score.sentence_bleu(true_sents, tree, smoothing_function=SmoothingFunction().method3))
        return acc
    def init_run(self, da):
        pass


def normalise(s):
    s = s.lower()
    words = s.split(" ")
    pos = nltk.pos_tag(words)
    result_words = []
    for word, tag in pos:
        if tag == 'NNS':
            if word == "children":
                result_words.append("child")
                result_words.append("-s")
                continue
            if not word.endswith("s"):
                print(word)
            result_words.append(word[:-1])
            result_words.append('-s')
        else:
            result_words.append(word)
    result = " ".join(result_words)
    return result


train = False
test_gen = True
test_res = False

if train:
    tf.reset_default_graph()
    da_train_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/tgen/e2e-challenge/input/train-das.txt"
    text_train_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/tgen/e2e-challenge/input/train-text.txt"

    seq2seq_config_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/tgen/e2e-challenge/config/config.yaml"
    seq2seq_model_file = "models/model_e2e_2/model.pickle.gz"
    parent_dir = os.path.abspath(os.path.join(seq2seq_model_file, os.pardir))
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)

    print('Training sequence-to-sequence generator...')
    config = Config(seq2seq_config_file)

    generator = Seq2SeqGen(config)

    generator.train(da_train_file, text_train_file)
    sys.setrecursionlimit(100000)
    generator.save_to_file(seq2seq_model_file)

if test_gen:
    for beam_size in [3]:
        tf.reset_default_graph()

        seq2seq_model_file = "models/model_e2e_2/model.pickle.gz"
        da_test_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/tgen/e2e-challenge/input/devel-das.txt"
        output_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/out-text-dir/no_rerank_b-{}.txt".format(beam_size)
        abstr_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/tgen/e2e-challenge/input/devel-abst.txt"
        parent_dir = os.path.abspath(os.path.join(output_file, os.pardir))
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)

        tgen = Seq2SeqBase.load_from_file(seq2seq_model_file)
        tgen.beam_size = beam_size
        tgen.classif_filter = None

        # read input files (DAs, contexts)
        das = read_das(da_test_file)
        # generate
        print('Generating...')
        gen_trees = []
        for num, da in enumerate(das, start=1):
            gen_trees.append(tgen.generate_tree(da))
        print(tgen.get_slot_err_stats())

        print('Lexicalizing...')
        tgen.lexicalize(gen_trees, abstr_file)

        print('Writing output...')
        if output_file.endswith('.txt'):
            gen_toks = [t.to_tok_list() for t in gen_trees]
            postprocess_tokens(gen_toks, das)
            write_tokens(gen_toks, output_file)

if test_res:
    for beam_size in [1, 3, 5, 10, 30, 100]:
        pred_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/out-text-dir/no_rerank_b-{}.txt".format(beam_size)
        true_file = "tgen/e2e-challenge/input/devel-text.txt"
        # true_file_2 = "bagel-data/data/training2/cv0{}/test.0-text.txt".format(cv)
        pred_sentences = []
        with open(pred_file, "r") as pred:
            for line in pred:
                pred_sentences.append(normalise(line.strip('\n')).split(" "))
        true_sentences = []
        with open(true_file, "r") as true:
            current_set = []
            for line in true:
                if len(line) > 1:
                    current_set.append(line.strip('\n').split(" "))
                else:
                    true_sentences.append(current_set)
                    current_set = []
        # with open(true_file_2, "r") as true_2:
        #     for line in true_2:
        #         true_sentences_2.append(normalise(line.strip('\n')).split(" "))

        acc = 0
        for t1, p in zip(true_sentences, pred_sentences):
            score = nltk.translate.bleu_score.sentence_bleu(t1, p,
                                                            smoothing_function=SmoothingFunction().method3)
            acc += score
            # print(score)
        print(beam_size, acc / len(true_sentences))
