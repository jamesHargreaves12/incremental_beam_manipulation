import regex as re
import matplotlib.pyplot as plt

from utils import get_true_sents, START_TOK, END_TOK


def parse_progress_file(filepath):
    beams = []
    steps = []
    hyps = []
    for line in open(filepath, 'r'):
        line = line.strip('\n')
        if not line:
            continue
        if re.match('Step: [0-9]+', line):
            if hyps:
                steps.append(hyps)
            hyps = []
        elif re.match('Test [0-9]+', line):
            if hyps:
                steps.append(hyps)
            if steps:
                beams.append(steps)
            steps = []
            hyps = []
        else:
            hyps.append(line)
    if hyps:
        steps.append(hyps)
    if steps:
        beams.append(steps)
    return beams


for beam_size in [3, 5, 10, 30]:
    progress_file_path = 'output_files/progress_files/{}.txt'.format(beam_size)
    beams = parse_progress_file(progress_file_path)
    true_texts = [[' '.join([START_TOK] + x + [END_TOK]) for x in xs] for xs in get_true_sents()]
    print(len(beams), len(true_texts))
    fall_out = []
    for cur_beams, true in zip(beams, true_texts):
        for i, beam in enumerate(cur_beams):
            if not any([any([x in y for y in true]) for x in beam]):
                fall_out.append(i)
                break
    plt.hist(fall_out, bins=max(fall_out) + 1)
    plt.title(beam_size)
    plt.show()
