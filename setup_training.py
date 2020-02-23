import json

with open("data/datasets/nl_to_sql/text2sql-data/data/advising.json", "r") as fp:
    data = json.load(fp)
acc_train = []
acc_test = []
acc_dev = []
for obj in data:
    if obj['query-split'] == 'train':
        # assert len(obj['sql']) == 1
        acc = acc_train
    elif obj['query-split'] == 'test':
        acc = acc_test
    else:
        acc = acc_dev

    sql = obj['sql'][0]  # for some sentence there are multiple outputs
    sentences = [x['text'] for x in obj['sentences']]
    for s in sentences:
        acc.append((s, sql))

with open("data/datasets/nl_to_sql/advising.train", "w+") as fp:
    for x, y in acc_train:
        fp.write("{} ||| {}\n".format(x, y))

with open("data/datasets/nl_to_sql/advising.dev", "w+") as fp:
    for x, y in acc_dev:
        fp.write("{} ||| {}\n".format(x, y))

with open("data/datasets/nl_to_sql/advising.test", "w+") as fp:
    for x, y in acc_test:
        fp.write("{} ||| {}\n".format(x, y))
