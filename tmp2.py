import pickle

from base_models import TGEN_Model
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from utils import get_training_variables, get_test_das
import numpy as np

texts, das = get_training_variables()
text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
da_embedder = DAEmbeddingSeq2SeqExtractor(das)

das_test = get_test_das()

beam_path = 'output_files/saved_beams/16_vanilla_2_3.pickle'
beam_path_2 = 'output_files/saved_beams/t2p_vanilla_3.pickle'
beams = pickle.load(file=open(beam_path, 'rb'))
beams_2 = pickle.load(file=open(beam_path_2, 'rb'))
models = TGEN_Model(da_embedder, text_embedder, 'new_configs/model_configs/seq2seq_all_data.yaml')
models.load_models()

da_emb = da_embedder.get_embeddings([das_test[0]])[0]
inf_enc_out = models.encoder_model.predict(np.array([da_emb]))
enc_outs = inf_enc_out[0]
enc_last_state = inf_enc_out[1:]

print(beams[0][0][1] == beams_2[0][0][1])
print(sum(models.get_prob_sequence(enc_outs, beams[0][0][1], enc_last_state)))
print(beams[0][0][0], ' '.join(text_embedder.reverse_embedding(beams[0][0][1])))
print()
print(sum(models.get_prob_sequence(enc_outs, beams_2[0][0][1], enc_last_state)))
print(beams_2[0][0][0], ' '.join(text_embedder.reverse_embedding(beams_2[0][0][1])))
print("********************************")
print(sum(models.get_prob_sequence(enc_outs, beams[0][1][1], enc_last_state)))
print(beams[0][1][0], ' '.join(text_embedder.reverse_embedding(beams[0][1][1])))
print()
print(sum(models.get_prob_sequence(enc_outs, beams_2[0][1][1], enc_last_state)))
print(beams_2[0][1][0], ' '.join(text_embedder.reverse_embedding(beams_2[0][1][1])))
print("********************************")
print(sum(models.get_prob_sequence(enc_outs, beams[0][2][1], enc_last_state)))
print(beams[0][2][0], ' '.join(text_embedder.reverse_embedding(beams[0][2][1])))
print()
print(sum(models.get_prob_sequence(enc_outs, beams_2[0][2][1], enc_last_state)))
print(beams_2[0][2][0], ' '.join(text_embedder.reverse_embedding(beams_2[0][2][1])))


print(models.get_prob_sequence(enc_outs, beams[0][0][1], enc_last_state))
