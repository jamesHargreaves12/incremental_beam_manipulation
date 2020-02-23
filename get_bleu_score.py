import nltk
from nltk.translate.bleu_score import SmoothingFunction

from normalise_data import normalise

pred_sentences = []
with open("tgen/out-text.txt", "r") as pred:
    for line in pred:
        pred_sentences.append(normalise(line).split(" "))
true_sentences = []
with open("tgen/bagel-data/data/training2/cv00/test.1-text.txt", "r") as true:
    for line in true:
        true_sentences.append(normalise(line).split(" "))
true_sentences_2 = []
with open("tgen/bagel-data/data/training2/cv00/test.0-text.txt", "r") as true_2:
    for line in true_2:
        true_sentences_2.append(normalise(line).split(" "))

acc = 0
for t1, t2, p in zip(true_sentences, true_sentences_2, pred_sentences):
    score = nltk.translate.bleu_score.sentence_bleu([t1, t2], p, smoothing_function=SmoothingFunction().method3)
    acc += score
    print(score)
print("average = ", acc / len(true_sentences))
