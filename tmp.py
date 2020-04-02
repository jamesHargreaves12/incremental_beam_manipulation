import sys
import os

from utils import get_true_sents, apply_absts

sys.path.append(os.path.join(os.getcwd(), 'tgen'))


from tgen.futil import smart_load_absts


test_abstr_file = "tgen/e2e-challenge/input/devel-abst.txt"
test_text_file = "tgen/e2e-challenge/input/devel-text.txt"
abstss = smart_load_absts(test_abstr_file)
texts = get_true_sents()

print(len(texts))
print(len(abstss))

texts_firsts = [t[0] for t in texts]

res = apply_absts(abstss,texts_firsts)
for r in res:
    print(" ".join(r))