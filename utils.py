import os
import sys

import nltk
import tensorflow as tf
from keras.engine.saving import model_from_yaml

sys.path.append(os.path.join(os.getcwd(), 'tgen'))
from enum import Enum
from tgen.logf import set_debug_stream


def construct_logs(beam_size):
    debug_stream = open("output_files/debug_files/output_gen_{}.txt".format(beam_size), "w+")
    set_debug_stream(debug_stream)


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


class RERANK(Enum):
    RANDOM = 0
    VANILLA = 1
    TGEN = 2
    ORACLE = 3
    REVERSE_ORACLE = 4

    def __str__(self):
        names = ["random", "vanilla", "tgen", "oracle", "reverse_oracle"]
        return names[self.value]


def get_true_sents():
    true_file = "tgen/e2e-challenge/input/devel-text.txt"

    true_sentences = []
    with open(true_file, "r") as true:
        current_set = []
        for line in true:
            if len(line) > 1:
                current_set.append(line.strip('\n').split(" "))
            else:
                true_sentences.append(current_set)
                current_set = []
    return true_sentences


def count_lines(filepath):
    return sum([1 for _ in open(filepath, "r").readlines()])


def get_truth_training():
    true_file_path = "tgen/e2e-challenge/input/train-text.txt"
    with open(true_file_path, "r") as fp:
        return [x.strip("\n") for x in fp.readlines()]


# def save_keras_model(model, file_path):
#     # serialize model to YAML
#     model_yaml = model.to_yaml()
#     with open(file_path, "w+") as yaml_file:
#         yaml_file.write(model_yaml)
#     # serialize weights to HDF5
#     weights_path = file_path.split(".")[0] + ".h5"
#     model.save_weights(weights_path)
#     print("Saved model to disk")
#
#
# def load_model(file_path):
#     yaml_file = open(file_path, 'r')
#     loaded_model_yaml = yaml_file.read()
#     yaml_file.close()
#     loaded_model = tf.keras.models.load_model(loaded_model_yaml)
#     # load weights into new mode
#     weights_path = file_path.split(".")[0] + ".h5"
#     loaded_model.load_weights(weights_path)
#     return loaded_model


def remove_strange_toks(tok):
    unimp_toks = ['<VOID>', '<UNK>', '<-s>']
    return '<STOP>' if tok in unimp_toks else tok

