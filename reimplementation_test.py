import random
import sys
import os
from time import time
import numpy as np

from base_model import TGEN_Model
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from utils import get_texts_training

sys.path.append(os.path.join(os.getcwd(), 'tgen'))
from tgen.futil import read_das

use_size = 100
valid_size = 20
epoch = 100
batch_size = 20
hidden_size = 128
embedding_size = 50
beam_size = 3
load_from_save = True

das = read_das("tgen/e2e-challenge/input/train-das.txt")

texts = [['<S>'] + x + ['<E>'] for x in get_texts_training()]
print(texts[0])

das = das[:use_size + valid_size]
texts = texts[:use_size + valid_size]

text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
da_embedder = DAEmbeddingSeq2SeqExtractor(das)

text_embs = text_embedder.get_embeddings(texts)
text_vsize = text_embedder.vocab_length
text_len = len(text_embs[0])

da_embs = da_embedder.get_embeddings(das)
da_vsize = da_embedder.vocab_length
da_len = len(da_embs[0])
print(da_vsize, text_vsize, da_len, text_len)

train_in = np.array(da_embs)
train_out = np.array(text_embs)

model_save_loc = "models/reimplementation"
models = TGEN_Model(batch_size, da_len, text_len, da_vsize, text_vsize, hidden_size, 50)
if load_from_save and os.path.exists(model_save_loc):
    models.load_models_from_location(model_save_loc)
else:
    models.train(train_in[:-valid_size], train_out[:-valid_size], epoch, train_in[-valid_size:], train_out[-valid_size:], text_embedder)
    models.save_model(model_save_loc)

# testing
test_das = read_das("tgen/e2e-challenge/input/devel-das.txt")
for beam_size in [1, 3, 5, 10, 30]:
    print("Beam_size {}".format(beam_size))
    start = time()
    with open("output_files/out-text-dir-v2/output_{}.txt".format(beam_size), "w+") as output_file:
        for da_emb in da_embedder.get_embeddings(test_das):
            pred = models.make_prediction(da_emb, text_embedder, beam_size)
            output_file.write(pred.replace(" <>", "") + "\n")
    print(time()-start)
    break

# full_model_save_path = 'models/reimp_save_test__.tf'
# if os.path.exists(full_model_save_path + '.index'):
#     print("Loading model from file")
#     full_model.load_weights(full_model_save_path)
# else:
#     print("Training model")
#     train(full_model, infer_enc_model, infer_dec_model, train_enc[:-valid_size], train_dec[:-valid_size], batch_size,
#           epoch, text_vsize, train_enc[-valid_size:], train_dec[-valid_size:], text_embedder)
# full_model.save_weights(full_model_save_path, save_format='tf')

# print("Testing greedy inference")
# encoder_in = train_enc[-valid_size]
# pred = make_prediction(encoder_in, infer_enc_model, infer_dec_model)
# print('Aim:    {}'.format(" ".join([x[0] for x in trees[-valid_size]])))
# print('Output: {}'.format(pred))
