import os

from e2e_metrics.metrics.pymteval import BLEUScore
from e2e_metrics.measure_scores import load_data

output_directory = 'output_files/out-text-dir-v3'


def test_res_official(pred_file_name):
    pred_file = os.path.join(output_directory, pred_file_name)
    true_file = "tgen/e2e-challenge/input/devel-conc.txt"
    data_src, data_ref, data_sys = load_data(true_file, pred_file)
    # mteval_scores = run_pymteval(data_ref, data_sys)

    bleu = BLEUScore()
    # print(data_ref[0])
    # print(data_sys[0])
    for sents_ref, sent_sys in zip(data_ref, data_sys):
        bleu.append(sent_sys, sents_ref)

    # return the computed scores
    bleu = bleu.score()

    return bleu


for filename in os.listdir(output_directory):
    print(filename, test_res_official(filename))
