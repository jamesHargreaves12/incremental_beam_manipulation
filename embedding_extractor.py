from utils import PAD_TOK, START_TOK, END_TOK
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'tgen'))
from tgen.data import DAI


class TokEmbeddingSeq2SeqExtractor(object):
    def __init__(self, tokenised_texts, max_length=None):
        self.vocab = set()
        for tokens in tokenised_texts:
            self.vocab.update(tokens)
        self.vocab.add(PAD_TOK)
        self.tok_to_embed = {tok: i for i, tok in enumerate(sorted(list(self.vocab)))}
        self.embed_to_tok = {i: tok for i, tok in enumerate(sorted(list(self.vocab)))}
        self.vocab_length = len(self.vocab)
        self.length = max([len(x) for x in tokenised_texts])
        if max_length is not None:
            self.length = min(self.length, max_length)
        self.start_emb = [self.tok_to_embed[START_TOK]]
        self.end_embs = [self.tok_to_embed[END_TOK], self.tok_to_embed[PAD_TOK]]
        self.empty_embedding = [self.tok_to_embed[PAD_TOK] for _ in range(self.length)]

    def get_embeddings(self, tokenised_texts, pad_from_end=True):
        embs = []
        for toks in tokenised_texts:
            emb = [self.tok_to_embed.get(x, self.tok_to_embed[PAD_TOK]) for x in toks]
            pad = [self.tok_to_embed[PAD_TOK] for _ in range(self.length - len(toks))]
            if pad_from_end:
                embs.append(emb + pad)
            else:
                embs.append(pad + emb)
        return [e[:self.length] for e in embs]

    def add_pad_to_embed(self, emb, to_start=False):
        pad = [self.tok_to_embed[PAD_TOK] for _ in range(self.length - len(emb))]
        if to_start:
            return pad + list(emb)
        else:
            return list(emb) + pad

    def reverse_embedding(self, embedding):
        return [self.embed_to_tok[e] for e in embedding]

    def remove_pad_from_embed(self, emb):
        return [x for x in emb if x != self.tok_to_embed[PAD_TOK]]

    def pad_to_length(self, text_emb, to_start=True):
        pads = [self.tok_to_embed[PAD_TOK]] * (self.length - len(text_emb))
        if to_start:
            return pads + text_emb
        else:
            return text_emb + pads


class DAEmbeddingSeq2SeqExtractor(object):
    UNK_ACT = 'UNK_ACT'
    UNK_VALUE = 'UNK_VALUE'
    UNK_SLOT = 'UNK_SLOT'

    def __init__(self, das):
        self.acts = {self.UNK_ACT}
        self.slots = {self.UNK_SLOT}
        self.values = {self.UNK_VALUE}
        self.coocurrence = set()
        for da in das:
            for dai in da:
                self.acts.add(dai.da_type)
                self.slots.add(dai.slot)
                self.values.add(dai.value)
                self.coocurrence.add((dai.da_type, dai.slot, dai.value))
        taken_emb = 0
        self.act_emb = {act: taken_emb + i for i, act in enumerate(sorted(list(self.acts)))}
        self.rev_act_emb = {taken_emb + i: act for i, act in enumerate(sorted(list(self.acts)))}
        taken_emb += len(self.acts)
        self.slot_emb = {slo: taken_emb + i for i, slo in enumerate(sorted(list(self.slots)))}
        self.rev_slot_emb = {taken_emb + i: slo for i, slo in enumerate(sorted(list(self.slots)))}
        taken_emb += len(self.slots)
        self.val_emb = {val: taken_emb + i for i, val in enumerate(sorted(list(self.values)))}
        self.rev_val_emb = {taken_emb + i: val for i, val in enumerate(sorted(list(self.values)))}
        taken_emb += len(self.values)
        self.vocab_length = taken_emb
        self.inclusion_map = {val: i for i, val in enumerate(sorted(list(self.coocurrence)))}
        self.inclusion_rev = {i: val for i, val in enumerate(sorted(list(self.coocurrence)))}
        self.length = max([len(x) for x in das]) * 3
        self.inclusion_length = len(self.inclusion_map)
        self.empty_inclusion = [0 for _ in range(self.inclusion_length)]
        self.empty_embedding = self.get_embeddings([[]])[0]

    def get_embeddings(self, das):
        embs = []
        unk_ack_emb = self.act_emb[self.UNK_ACT]
        unk_slot_emb = self.slot_emb[self.UNK_SLOT]
        unk_val_emb = self.val_emb[self.UNK_VALUE]
        for da in das:
            emb = []
            for dai in sorted(da):
                emb.append(self.act_emb.get(dai.da_type, unk_ack_emb))
                emb.append(self.slot_emb.get(dai.slot, unk_slot_emb))
                emb.append(self.val_emb.get(dai.value, unk_val_emb))

            pad = [unk_ack_emb, unk_slot_emb, unk_val_emb] * (self.length // 3 - len(da))
            embs.append(pad + emb)
        return [e[-self.length:] for e in embs]

    def reverse_embedding(self, da_emb):
        i = 0
        das = []
        while i < len(da_emb):
            act = self.rev_act_emb[da_emb[i]]
            slot = self.rev_slot_emb[da_emb[i + 1]]
            val = self.rev_val_emb[da_emb[i + 2]]
            i += 3
            if act != self.UNK_ACT:
                das.append(DAI(act, slot, val))
        return das

    def remove_pad_from_embed(self, emb):
        results = []
        for i in range(0, len(emb), 3):
            if emb[i] == self.act_emb[self.UNK_ACT] and emb[i + 1] == self.slot_emb[self.UNK_SLOT] and emb[i + 2] == \
                    self.val_emb[self.UNK_VALUE]:
                continue
            results.append(emb[i])
            results.append(emb[i + 1])
            results.append(emb[i + 2])
        return results

    def add_pad_to_embed(self, emb, to_start=True):
        pad = [self.act_emb[self.UNK_ACT], self.slot_emb[self.UNK_SLOT], self.val_emb[self.UNK_VALUE]] \
              * ((self.length - len(emb))//3)
        if to_start:
            return pad+list(emb)
        else:
            return list(emb)+pad

    def get_inclusion(self, das):
        included = set()
        for dai in das:
            if (dai.da_type, dai.slot, dai.value) in self.inclusion_map:
                included.add(self.inclusion_map[(dai.da_type, dai.slot, dai.value)])
        return [(1 if x in included else 0) for x in range(self.inclusion_length)]

    def reverse_inclusion(self, inclusion):
        included = set()
        for i, x in enumerate(inclusion):
            if x == 1:
                included.add(i)
        return [self.inclusion_rev[x] for x in included]


if __name__ == "__main__":
    texts = ["<S> My name is James . <E>", "<S> My favourite food is steak . <E>", "<S> A <E>"]
    tokenized = [x.split(" ") for x in texts]
    print(tokenized)
    emb = TokEmbeddingSeq2SeqExtractor(tokenized)
    print(emb.get_embeddings(tokenized))
    print(emb.get_embeddings([['<S>']]))
