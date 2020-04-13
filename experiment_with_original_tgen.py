import os
import re
import sys

import nltk
import yaml

from utils import get_test_das, get_final_beam, START_TOK, PAD_TOK, END_TOK

sys.path.append(os.path.join(os.getcwd(), 'tgen'))

from tgen.tfclassif import RerankingClassifier, Reranker


def get_fake_node(val):
    if val != START_TOK:
        t_lemma = val
        formeme = 'x'
    else:
        t_lemma = None
        formeme = None
    return t_lemma, formeme


class fake_trees(object):
    def __init__(self, hyp):
        self.nodes = [get_fake_node(x) for x in hyp if x not in [END_TOK, PAD_TOK]]


test_das = get_test_das()
classif_filter = Reranker.load_from_file('tgen/models/model_e2e_2/model.tftreecl.pickle.gz')
for beam_size in [3, 5, 10, 30, 100]:
    final_beams = get_final_beam(beam_size)
    out_file = open('output_files/out-text-dir-v3/TGEN_b-{}.txt'.format(beam_size), "w+")
    for beam, das in zip(final_beams, test_das):
        ftrees = [fake_trees(x[0]) for x in beam]
        classif_filter.init_run(das)
        fits = classif_filter.dist_to_cur_da(ftrees)
        scores = []
        for fit, hyp_score in zip(fits, beam):
            scores.append((-100 * fit + hyp_score[1], hyp_score[0]))
        best = sorted(scores, reverse=True)[0][1]
        out_file.write(" ".join(best) + '\n')
    out_file.close()
