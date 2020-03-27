from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

from utils import get_truth_training

texts = [["<GO>"]+x.split(" ")+["<STOP>"] for x in get_truth_training()]

print(texts[0], len(texts))
path = get_tmpfile("models/word2vec.model")
size_embed = 30
model = Word2Vec(texts, size=size_embed, window=5, min_count=1, workers=4)
model.save("models/word2vec_{}.model".format(size_embed))
