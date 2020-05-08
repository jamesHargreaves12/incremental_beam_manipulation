import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from base_models import TGEN_Reranker, TGEN_Model
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from utils import get_training_variables, get_hamming_distance

cfg_path = "new_configs/model_configs/tgen-seq2seq.yaml"
cfg = yaml.load(open(cfg_path, "r"))
texts, das, = get_training_variables()
text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
da_embedder = DAEmbeddingSeq2SeqExtractor(das)
train_text = np.array(text_embedder.get_embeddings(texts, pad_from_end=True) + [text_embedder.empty_embedding])

da_embs = da_embedder.get_embeddings(das) + [da_embedder.empty_embedding]
seq2seq = TGEN_Model(da_embedder, text_embedder, cfg_path)
seq2seq.train(da_seq=np.array(da_embs),
              text_seq=np.array(train_text),
              n_epochs=cfg["epoch"],
              valid_size=cfg["valid_size"],
              early_stop_point=cfg["min_epoch"],
              minimum_stop_point=0)
