import os
import re
import sys

import nltk

sys.path.append(os.path.join(os.getcwd(), 'tgen'))
from enum import Enum
from tgen.logf import set_debug_stream
from tgen.futil import read_das
from tgen.futil import smart_load_absts

START_TOK = '<S>'
END_TOK = '<E>'
PAD_TOK = '<>'


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


def get_texts_training():
    true_file_path = "tgen/e2e-challenge/input/train-text.txt"
    with open(true_file_path, "r") as fp:
        return [x.strip("\n").split(" ") for x in fp.readlines()]


def apply_absts(absts, texts):
    results = []
    pattern = re.compile("X-[a-z]+")
    for abst, text in zip(absts, texts):
        text_res = []
        # print(text)
        for tok in text:
            if pattern.match(tok):
                slot = tok[2:]
                for a in abst:
                    if a.slot == slot:
                        text_res.append(a.value)
                        break
            else:
                text_res.append(tok)
        # assert(len(text) == len(text_res))
        results.append(text_res)
    return results


def get_training_das_texts():
    das = read_das("tgen/e2e-challenge/input/train-das.txt")
    texts = [[START_TOK] + x + [END_TOK] for x in get_texts_training()]
    return das, texts


def safe_get_w2v(w2v, tok):
    unimp_toks = [PAD_TOK]
    tok = END_TOK if tok in unimp_toks else tok
    return w2v[tok]


def remove_strange_toks(tok):
    unimp_toks = ['<VOID>', '<UNK>', '<-s>']
    return '<STOP>' if tok in unimp_toks else tok


def get_hamming_distance(xs, ys):
    return sum([1 for x, y in zip(xs, ys) if x != y])


def get_training_variables():
    das = read_das("tgen/e2e-challenge/input/train-das.txt")
    texts = [[START_TOK] + x + [END_TOK] for x in get_texts_training()]
    return texts, das


def get_test_das():
    das = read_das("tgen/e2e-challenge/input/devel-das.txt")
    return das


def get_abstss():
    return smart_load_absts('tgen/e2e-challenge/input/train-abst.txt')


def get_final_beam(beam_size):
    path = "output_files/saved_beams/vanilla_{}.txt".format(beam_size)
    output= []
    current = []
    for line in open(path, "r+"):
        if line == '\n':
            output.append(current)
            current = []
            continue

        toks = line.strip('\n').split(" ")
        logprob = float(toks.pop())
        current.append((toks, logprob))
    return output
