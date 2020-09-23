import os

import yaml


optimization_configs = os.path.join('new_configs/optimization_configs')
filenames = os.listdir(optimization_configs)
filepaths = [os.path.join(optimization_configs, filename) for filename in filenames]
cfg = None
for fp in filepaths:
    cfg = yaml.safe_load(open(fp, 'r'))
    if cfg['state'] == 'created':
        break
if not cfg or cfg['state'] != 'created':
    print('No remaining Configs')
    exit(0)
else:
    cfg['state'] = 'running'
    yaml.dump(cfg, open(fp, 'w+'))

print("Using config from: {}".format(fp))
if "trainable_reranker_config" in cfg:
    cfg["train_reranker"] = yaml.safe_load(open(cfg["trainable_reranker_config"], "r"))
print("Config:")
[print("\t{}: {}".format(k, v)) for k, v in cfg.items()]
print("*******")

from base_models import TGEN_Model
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from get_results import do_beam_search
from get_results_bleu_scores import print_results
from utils import CONFIGS_DIR, get_training_variables, get_test_das, get_multi_reference_training_variables, \
    get_true_sents, get_abstss_test

texts, das = get_training_variables()
text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
da_embedder = DAEmbeddingSeq2SeqExtractor(das)

das_test = get_test_das()
if "get_train_beam" in cfg and cfg["get_train_beam"]:
    _, das_test = get_multi_reference_training_variables()

true_vals = get_true_sents()
models = TGEN_Model(da_embedder, text_embedder, cfg['tgen_seq2seq_config'])
models.load_models()

if cfg.get("first_x", False):
    das_test = das_test[:cfg['first_x']]

absts = get_abstss_test()
for beam_size in cfg["beam_sizes"]:
    do_beam_search(beam_size, cfg, models, das_test, da_embedder, text_embedder, true_vals, absts)

print_results()
