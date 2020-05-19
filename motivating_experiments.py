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
    progress_file_path = 'output_files/progress_files/vanilla_{}.txt'.format(beam_size)
    beams = parse_progress_file(progress_file_path)
    true_texts = [[' '.join([START_TOK] + x + [END_TOK]) for x in xs] for xs in get_true_sents()]
    if len(beams) != len(true_texts):
        continue
    fall_out = []
    lens = []
    for cur_beams, true in zip(beams, true_texts):
        best = 0
        for t in true:
            lens.append(len(t.split(" ")))
            for i, beam in enumerate(cur_beams):
                if not any([x in t for x in beam]):
                    best = max(best, i / len(t.split(" ")))
                    # fall_out.append()
                    break
            else:
                best = 1
                break
        fall_out.append(best)
    arr = plt.hist(fall_out, bins=50)
    print(arr)
    print(sum(lens)/len(lens))
    plt.title(progress_file_path.split('/')[-1])
    plt.xlabel("Proportion generated before fell out")
    # plt.ylabel("")
    # plt.ylim(0, 160*)
    # for i in range(max(fall_out) + 1):
    #     plt.text(arr[1][i], arr[0][i], str(arr[0][i]))

    plt.show()
