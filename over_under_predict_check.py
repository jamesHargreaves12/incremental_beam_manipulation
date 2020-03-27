from e2e_metrics.measure_scores import load_data
import matplotlib.pyplot as plt

from utils import RERANK


def get_pred_true_length(reranker, beam_size):
    pred_file = "out-text-dir-v2/{}_b-{}.txt".format(reranker,
                                                  beam_size)
    true_file = "tgen/e2e-challenge/input/devel-text.txt"
    data = load_data(true_file, pred_file)
    data_src, data_ref, data_sys = data
    acc = []
    for pred, ref in zip(data_sys, data_ref):
        ref_len = sum([len(r.split(" ")) for r in ref])/len(ref)
        pred_len = len(pred.split(" "))
        acc.append((pred_len, ref_len))
    return acc


if __name__ == "__main__":
    for beam_size in [3, 5, 10, 30, 100]:
        results = get_pred_true_length(RERANK.TGEN, beam_size)

        plt.scatter([x[1] for x in results], [x[0] for x in results], alpha=0.25)
        plt.title(beam_size)
        plt.plot([0, max(x[1] for x in results)], [0, max(x[1] for x in results)], 'k-',
                 color='red')
        plt.xlabel("Length of true reference")
        plt.ylabel("Length of prediction")
        plt.show()
