# ./run_tgen.py seq2seq_train
import os
import sys
from getopt import getopt
sys.path.append(os.path.join(os.getcwd(),'tgen'))

import nltk
from nltk.translate.bleu_score import SmoothingFunction
from tgen.config import Config
from tgen.seq2seq import Seq2SeqGen
import tensorflow as tf

from tgen.seq2seq import Seq2SeqBase

from tgen.futil import read_das, write_ttrees, write_tokens, postprocess_tokens, create_ttree_doc

def normalise(s):
    s = s.lower()
    words = s.split(" ")
    pos = nltk.pos_tag(words)
    x =1
    result_words = []
    for word, tag in pos:
        if tag == 'NNS':
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
train_start = 0
rerank = True

if train:

    for cv in range(train_start, 10):
        tf.reset_default_graph()
        da_train_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/tgen/bagel-data/input/cv0{}/train-das.txt".format(
            cv)
        text_train_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/tgen/bagel-data/input/cv0{}/train-text.txt".format(
            cv)

        seq2seq_config_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/tgen/bagel-data/config/seq2seq.py"
        seq2seq_model_file = "models/model_{}/model.pickle.gz".format(cv)
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
    for cv in range(10):
        for beam_size in [1, 3, 5, 10, 30, 100]:
            tf.reset_default_graph()

            seq2seq_model_file = "models/model_{}/model.pickle.gz".format(cv)
            da_test_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/tgen/bagel-data/data/training2/cv0{}/test-das.txt".format(cv)
            output_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/tgen/out-text-dir/cv-{}/b-{}.txt".format(
                cv, beam_size)
            parent_dir = os.path.abspath(os.path.join(output_file, os.pardir))
            if not os.path.exists(parent_dir):
                os.mkdir(parent_dir)

            tgen = Seq2SeqBase.load_from_file(seq2seq_model_file)
            tgen.beam_size = beam_size

            # read input files (DAs, contexts)
            das = read_das(da_test_file)
            # generate
            print('Generating...')
            gen_trees = []
            for num, da in enumerate(das, start=1):
                gen_trees.append(tgen.generate_tree(da))
            print(tgen.get_slot_err_stats())

            # write output .yaml.gz or .txt
            if output_file is not None:
                print('Writing output...')
                if output_file.endswith('.txt'):
                    gen_toks = [t.to_tok_list() for t in gen_trees]
                    postprocess_tokens(gen_toks, das)
                    write_tokens(gen_toks, output_file)

if test_res:
    for beam_size in [1, 3, 5, 10, 30, 100]:
        for cv in range(10):
            pred_file = "/Users/james_hargreaves/PycharmProjects/partIII-Project/tgen/out-text-dir/cv-{}/b-{}.txt".format(
                cv, beam_size)
            true_file = "bagel-data/data/training2/cv0{}/test.1-text.txt".format(cv)
            true_file_2 = "bagel-data/data/training2/cv0{}/test.0-text.txt".format(cv)
            pred_sentences = []
            with open(pred_file, "r") as pred:
                for line in pred:
                    pred_sentences.append(normalise(line.strip('\n')).split(" "))
            true_sentences = []
            with open(true_file, "r") as true:
                for line in true:
                    true_sentences.append(normalise(line.strip('\n')).split(" "))
            true_sentences_2 = []
            with open(true_file_2, "r") as true_2:
                for line in true_2:
                    true_sentences_2.append(normalise(line.strip('\n')).split(" "))

            acc = 0
            for t1, t2, p in zip(true_sentences, true_sentences_2, pred_sentences):
                score = nltk.translate.bleu_score.sentence_bleu([t1, t2], p,
                                                                smoothing_function=SmoothingFunction().method3)
                acc += score
                # print(score)
            print(beam_size, cv, acc / len(true_sentences))
