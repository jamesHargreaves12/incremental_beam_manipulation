import nltk

s = 'I am sorry but there are no venues near X in the city centre .'

def normalise(s):
    s = s.lower()
    words = s.split(" ")
    pos = nltk.pos_tag(words)
    x =1
    result_words = []
    for word, tag in pos:
        if tag == 'NNS':
            if not word.endswith("s"):
                print(word)
            result_words.append(word[:-1])
            result_words.append('-s')
        else:
            result_words.append(word)
    result = " ".join(result_words)
    return result

if __name__ == '__main__':
    with open('tgen/bagel-data/input/all-text-old.txt','r') as input_fp:
        with open('tgen/bagel-data/input/all-text-norm.txt','w') as output_fp:
            for line in input_fp:
                sent = line.strip('\n')
                output_fp.write(normalise(sent) + "\n")
