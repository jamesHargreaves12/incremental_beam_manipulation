import os

from e2e_metrics.metrics.pymteval import BLEUScore
from e2e_metrics.measure_scores import load_data

output_directory = 'output_files/from_gpu/out-text-dir-v3'
# output_directory = 'output_files/out-text-dir-v3'


def test_res_official(pred_file_name):
    pred_file = os.path.join(output_directory, pred_file_name)
    true_file = "tgen/e2e-challenge/input/devel-conc.txt"
    data_src, data_ref, data_sys = load_data(true_file, pred_file)
    # mteval_scores = run_pymteval(data_ref, data_sys)

    bleu = BLEUScore()
    for sents_ref, sent_sys in zip(data_ref, data_sys):

        bleu.append(sent_sys, sents_ref)

    # return the computed scores
    bleu = bleu.score()

    return bleu


filename_bs = []
for filename in os.listdir(output_directory):
    beam_size = int("".join([x for x in filename if x.isdigit()]))
    filename_bs.append((filename, beam_size))

for filename, bs in sorted(filename_bs, key=lambda x: (x[0][:5], x[1])):
    print(filename, bs, test_res_official(filename))
