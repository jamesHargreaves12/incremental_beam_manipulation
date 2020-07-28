import datetime
import os
import sys
import time

from e2e_metrics.metrics.pymteval import BLEUScore
from e2e_metrics.measure_scores import load_data
from utils import RESULTS_DIR, VALIDATION_NOT_TEST


def test_res_official(pred_file_name):
    pred_file = os.path.join(RESULTS_DIR, pred_file_name)
    if VALIDATION_NOT_TEST:
        true_file = "tgen/e2e-challenge/input/devel-conc.txt"
    else:
        true_file = "tgen/e2e-challenge/input/test-conc.txt"
    _, data_ref, data_sys = load_data(true_file, pred_file)

    bleu = BLEUScore()
    for sents_ref, sent_sys in zip(data_ref, data_sys):
        bleu.append(sent_sys, sents_ref)

    total_bleu_score = bleu.score()
    bleu_scores = []
    # for sents_ref, sent_sys in zip(data_ref, data_sys):
    #     bleu.reset()
    #     bleu.append(sent_sys, sents_ref)
    #     bleu_scores.append(bleu.score())
    # # return the computed scores
    # if total_bleu_score > 0.6:
    #     print(bleu_scores)

    return total_bleu_score


def print_results():
    day_seconds = 24 * 60 * 60
    print(sys.argv)
    filename_bs = []
    for filename in os.listdir(RESULTS_DIR):
        if '*'in filename:
            continue
        splits = filename.split('-')
        beam_size = int(splits[-1].split('.')[0])
        filter_name = '-'.join(splits[:-1])
        if (len(sys.argv) > 1 and sys.argv[1] == 'all') or os.path.getmtime(
                os.path.join(RESULTS_DIR, filename)) > time.time() - day_seconds / 2:
            filename_bs.append((filter_name, filename, beam_size))

    for _, filename, bs in sorted(filename_bs, key=lambda x: (x[0], int(x[2]))):
        print(filename, bs, test_res_official(filename))


if __name__ == "__main__":
    # RESULTS_DIR = 'output_files/from_gpu_2/out-text-dir-v3'
    print_results()
