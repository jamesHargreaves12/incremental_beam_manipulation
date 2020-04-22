import yaml
import numpy as np

from base_model import TrainableReranker
from e2e_metrics.metrics.pymteval import BLEUScore
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from utils import get_final_beam, get_training_variables, get_abstss_train, PAD_TOK, END_TOK, START_TOK
from matplotlib import pyplot as plt
cfg = yaml.load(open("configs/config_trainable_reranker_train.yaml", "r+"))
final_beams = get_final_beam(3, True)
texts, das, = get_training_variables()
text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
da_embedder = DAEmbeddingSeq2SeqExtractor(das)
bleu_scorer = BLEUScore()
input_text = []
input_da = []
input_score = []

for beam, text, da in zip(final_beams, texts, das):
    for hyp in beam:
        pred_toks = [x for x in hyp[0] if x not in [START_TOK, END_TOK, PAD_TOK]]
        true_toks = [x for x in text if x not in [START_TOK, END_TOK, PAD_TOK]]
        bleu_scorer.reset()
        bleu_scorer.append(pred_toks, [text])
        score = bleu_scorer.score()
        input_text.append(hyp[0])
        input_da.append(da)
        input_score.append(score)

print(input_score[0])
plt.hist(input_score)
plt.show()

reranker = TrainableReranker(da_embedder, text_embedder, cfg)
da_embs = np.array(da_embedder.get_embeddings(input_da))
text_embs = np.array(text_embedder.get_embeddings(input_text))
bleu_scores = np.array(input_score).reshape(-1, 1)
valid_size = cfg['valid_size']
reranker.train(text_embs[:-valid_size], da_embs[:-valid_size], bleu_scores[:-valid_size], 10, text_embs[-valid_size:], da_embs[-valid_size:], bleu_scores[-valid_size:])
# reranker.load_model()

x= 1