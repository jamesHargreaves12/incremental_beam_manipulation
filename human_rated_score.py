from e2e_metrics.measure_scores import load_data
from e2e_metrics.metrics.pymteval import BLEUScore
import matplotlib.pyplot as plt

from utils import RERANK

true_file = "tgen/e2e-challenge/input/devel-text.txt"
pred_file = "output_files/out-text-dir-v2/{}_b-{}.txt".format(RERANK.ORACLE, 5)
data_src, data_ref, data_sys = load_data(true_file, pred_file)

acc_human = []
acc_oracle = []
bleu = BLEUScore()
bleu = bleu.score()

for sents_ref, sent_sys in zip(data_ref, data_sys):
    bleu = BLEUScore()
    bleu.append(sents_ref[0],sents_ref[1:])
    acc_human.append(bleu.score())
    bleu = BLEUScore()
    bleu.append(sent_sys,sents_ref[1:])
    acc_oracle.append(bleu.score())

plt.hist(acc_human, bins=50, alpha=0.5, label='human')
plt.hist(acc_oracle, bins=50, alpha=0.5, label='oracle_5')
plt.xlabel("Bleu Score")
plt.ylabel("Frequency")
plt.show()

# 0.5817323397723214
# oracle score for the same references
# [(3, 0.574925505832616), (5, 0.5989089524462489), (10, 0.636601292189156), (30, 0.6990794985388977), (100, 0.72675704978443)]