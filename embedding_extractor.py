class TokEmbeddingSeq2SeqExtractor(object):
    PAD_TOK = '<>'

    def __init__(self, tokenised_texts, max_length=None):
        self.vocab = set()
        for tokens in tokenised_texts:
            self.vocab.update(tokens)
        self.vocab.add(self.PAD_TOK)
        self.tok_to_embed = {tok: i for i, tok in enumerate(sorted(list(self.vocab)))}
        self.embed_to_tok = {i: tok for i, tok in enumerate(sorted(list(self.vocab)))}
        self.vocab_length = len(self.vocab)
        self.length = max([len(x) for x in tokenised_texts])
        if max_length is not None:
            self.length = min(self.length, max_length)

    def get_embeddings(self, tokenised_texts):
        embs = []
        for toks in tokenised_texts:
            emb = [self.tok_to_embed.get(x, self.PAD_TOK) for x in toks]
            pad = [self.tok_to_embed['<>'] for _ in range(self.length - len(toks))]
            embs.append(emb + pad)
        return [e[:self.length + 1] for e in embs]


class DAEmbeddingSeq2SeqExtractor(object):
    UNK_ACT = 'UNK_ACT'
    UNK_VALUE = 'UNK_VALUE'
    UNK_SLOT = 'UNK_SLOT'

    def __init__(self, das):
        self.acts = {self.UNK_ACT}
        self.slots = {self.UNK_SLOT}
        self.values = {self.UNK_VALUE}
        for da in das:
            for dai in da:
                self.acts.add(dai.da_type)
                self.slots.add(dai.slot)
                self.values.add(dai.value)
        taken_emb = 0
        self.act_emb = {act: taken_emb + i for i, act in enumerate(sorted(list(self.acts)))}
        taken_emb += len(self.acts)
        self.slot_emb = {slo: taken_emb + i for i, slo in enumerate(sorted(list(self.slots)))}
        taken_emb += len(self.slots)
        self.val_emb = {val: taken_emb + i for i, val in enumerate(sorted(list(self.values)))}
        taken_emb += len(self.values)
        self.vocab_length = taken_emb
        self.length = max([len(x) for x in das])

    def get_embeddings(self, das):
        embs = []
        for da in das:
            emb = []
            for dai in sorted(da):
                emb.append(self.act_emb.get(dai.da_type, self.UNK_ACT))
                emb.append(self.slot_emb.get(dai.slot, self.UNK_SLOT))
                emb.append(self.val_emb.get(dai.value, self.UNK_VALUE))

            pad = [self.act_emb[self.UNK_ACT], self.slot_emb[self.UNK_SLOT], self.val_emb[self.UNK_VALUE]] \
                  * (self.length - len(da))
            embs.append(pad + emb)
        return [e[-self.length*3:] for e in embs]


if __name__ == "__main__":
    texts = ["<S> My name is James . <E>", "<S> My favourite food is steak . <E>", "<S> A <E>"]
    tokenized = [x.split(" ") for x in texts]
    print(tokenized)
    emb = TokEmbeddingSeq2SeqExtractor(tokenized)
    print(emb.get_embeddings(tokenized))
    print(emb.get_embeddings([['<S>']]))
