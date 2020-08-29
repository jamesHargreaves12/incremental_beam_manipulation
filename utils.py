import os
import re
import sys
from collections import defaultdict
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

import h5py
import nltk
from keras.engine.saving import load_weights_from_hdf5_group
import numpy as np
import json

sys.path.append(os.path.join(os.getcwd(), 'tgen'))
from enum import Enum
from tgen.futil import read_das
from tgen.futil import smart_load_absts
from regex import Regex, UNICODE, IGNORECASE

START_TOK = '<S>'
END_TOK = '<E>'
PAD_TOK = '<>'
RESULTS_DIR = 'output_files/out-text-dir-v3'
CONFIGS_DIR = 'new_configs'
CONFIGS_MODEL_DIR = 'new_configs/model_configs'
TRAIN_BEAM_SAVE_FORMAT = 'output_files/saved_beams/train_vanilla_{}_{}.pickle'
TEST_BEAM_SAVE_FORMAT = 'output_files/saved_beams/vanilla_{}.pickle'
VALIDATION_NOT_TEST= True
DATASET_WEBNLG=True


class fakeDAI:
    def __init__(self, triple):
        self.slot, self.da_type, self.value = triple

    def __lt__(self, other):
        return (self.slot, self.da_type, self.value) < (other.slot, other.da_type, other.value)


def get_das_texts_from_webnlg(filepath):
    json_data = json.load(open(filepath, 'r'));
    das = []
    tokss = []
    for item in json_data:
        ent2ner = {v: k for k, v in item['ner2ent'].items()}
        lexicalized_triples = [[ent2ner.get(t, t) for t in trip] for trip in item['triples']]

        da = [fakeDAI(trip) for trip in lexicalized_triples]
        toks = [START_TOK] + item['target'].split(' ') + [END_TOK]
        das.append(da)
        tokss.append(toks)
    return das, tokss


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
    if DATASET_WEBNLG:
        if VALIDATION_NOT_TEST:
            return get_das_texts_from_webnlg("WebNLG_Reader/data/webnlg/valid.json")[1]
        else:
            return get_das_texts_from_webnlg("WebNLG_Reader/data/webnlg/test.json")[1]


    if VALIDATION_NOT_TEST:
        true_file = "tgen/e2e-challenge/input/devel-text.txt"
    else:
        true_file = "tgen/e2e-challenge/input/test-text.txt"

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
    if DATASET_WEBNLG:
        return get_das_texts_from_webnlg('WebNLG_Reader/data/webnlg/train.json')
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
    if DATASET_WEBNLG:
        das,texts = get_das_texts_from_webnlg('WebNLG_Reader/data/webnlg/train.json')
        return texts,das
    das = read_das("tgen/e2e-challenge/input/train-das.txt")
    texts = [[START_TOK] + x + [END_TOK] for x in get_texts_training()]
    return texts, das


def get_multi_reference_training_variables():
    texts, das = get_training_variables()

    da_text_map = defaultdict(list)
    for da, text in zip(das, texts):
        da_text_map[tuple(da)].append(text)
    das_mr = []
    texts_mr = []
    for da, text in da_text_map.items():
        das_mr.append(da)
        texts_mr.append(text)
    return texts_mr, das_mr


def get_test_das():
    if DATASET_WEBNLG:
        if VALIDATION_NOT_TEST:
            return get_das_texts_from_webnlg("WebNLG_Reader/data/webnlg/valid.json")[0]
        else:
            return get_das_texts_from_webnlg("WebNLG_Reader/data/webnlg/test.json")[0]

    if VALIDATION_NOT_TEST:
        das_file = "tgen/e2e-challenge/input/devel-das.txt"
    else:
        das_file = "tgen/e2e-challenge/input/test-das.txt"

    das = read_das(das_file)
    return das


def get_abstss_train():
    return smart_load_absts('tgen/e2e-challenge/input/train-abst.txt')


def get_abstss_test():
    if VALIDATION_NOT_TEST:
        absts_file = 'tgen/e2e-challenge/input/devel-abst.txt'
    else:
        absts_file = 'tgen/e2e-challenge/input/test-abst.txt'

    return smart_load_absts(absts_file)


def get_final_beam(beam_size, train=False):
    if train:
        path_format = "output_files/saved_beams/train_vanilla_{}.txt"
    else:
        path_format = "output_files/saved_beams/vanilla_{}.txt"
    path = path_format.format(beam_size)
    output = []
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


def load_model_from_gpu(model, filepath):
    f = h5py.File(filepath, mode='r')
    load_weights_from_hdf5_group(f['model_weights'], model.layers)


def get_features(path, text_embedder, w2v, tok_prob):
    h = path[2][0][0]
    c = path[2][1][0]
    pred_words = [text_embedder.embed_to_tok[x] for x in path[1]]

    return np.concatenate((h, c,
                           safe_get_w2v(w2v, pred_words[-1]), safe_get_w2v(w2v, pred_words[-2]),
                           [tok_prob, path[0], len(pred_words)]))


def postprocess(text):
    text = re.sub(r'([a-zA-Z]) - ([a-zA-Z])', r'\1-\2', text)
    text = re.sub(r'£ ([0-9])', r'£\1', text)
    text = re.sub(r'([a-zA-Z]) *\' *(s|m|d|ll|re|ve|t)', r"\1'\2", text)
    text = re.sub(r'([a-zA-Z]) *n\'t', r"\1n't", text)
    text = re.sub(r' \' ([a-zA-Z ]+) \' ', r" '\1' ", text)
    return text


def tgen_postprocess(text):
    currency_or_init_punct = Regex(r' ([\p{Sc}\(\[\{\¿\¡]+) ', flags=UNICODE)
    noprespace_punct = Regex(r' ([\,\.\?\!\:\;\\\%\}\]\)]+) ', flags=UNICODE)
    contract = Regex(r" (\p{Alpha}+) ' (ll|ve|re|[dsmt])(?= )", flags=UNICODE | IGNORECASE)
    dash_fixes = Regex(r" (\p{Alpha}+|£ [0-9]+) - (priced|star|friendly|(?:£ )?[0-9]+) ",
                       flags=UNICODE | IGNORECASE)
    dash_fixes2 = Regex(r" (non) - ([\p{Alpha}-]+) ", flags=UNICODE | IGNORECASE)

    text = ' ' + text + ' '
    text = dash_fixes.sub(r' \1-\2 ', text)
    text = dash_fixes2.sub(r' \1-\2 ', text)
    text = currency_or_init_punct.sub(r' \1', text)
    text = noprespace_punct.sub(r'\1 ', text)
    text = contract.sub(r" \1'\2", text)
    text = text.strip()
    # capitalize
    if not text:
        return ''
    text = text[0].upper() + text[1:]
    return text


def get_regression_vals(num_ranks, with_train_refs):
    if with_train_refs:
        return [i / num_ranks for i in range(1, num_ranks + 1)]
    else:
        return [i / (num_ranks - 1) for i in range(num_ranks)]


def get_section_cutoffs(num_ranks):
    return [i / num_ranks for i in range(1, num_ranks)]


def get_section_value(val, cut_offs, regression_vals, merge_middle=False, only_top=False, only_bottom=False):
    def group_sections(x):
        if merge_middle:
            return 1 if x > 0.999 else (0 if x < 0.001 else 0.5)
        elif only_bottom:
            return 1 if x > 0.999 else 0
        elif only_top:
            return 0 if x < 0.001 else 1
        else:
            return x

    for i, co in enumerate(cut_offs):
        if val <= co:
            return group_sections(regression_vals[i])
    return group_sections(1)
