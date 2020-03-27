import os
import pickle
import random
import sys

import nltk
from gensim.models import Word2Vec
from nltk.translate.bleu_score import SmoothingFunction

from e2e_metrics.metrics.pymteval import BLEUScore
from utils import RERANK, remove_strange_toks

sys.path.append(os.path.join(os.getcwd(), 'tgen'))

from future.utils import old_div
from tgen.tree import TreeData, NodeData
from tgen.tfclassif import Reranker
from tgen.config import Config
from tgen.seq2seq import Seq2SeqGen, Seq2SeqBase, cut_batch_into_steps
from tgen.logf import log_info, log_debug, log_warn
from tgen.futil import read_das, write_ttrees, write_tokens, postprocess_tokens, create_ttree_doc

import numpy as np
import tensorflow as tf

from tgen.futil import smart_load_absts

training_data = []
w2v_model = None
bc_model = None


def save_training_data(filepath):
    with open(filepath, "w+") as fp:
        for training in training_data:
            fp.write(",".join([str(x) for x in training]) + '\n')


def lexicalize_beam(lexicalizer, gen_trees, absts):
    """Lexicalize nodes in the generated trees (which may represent trees, tokens, or tagged lemmas).
    Expects lexicalization file (and surface forms file) to be loaded in the Lexicalizer object,
    otherwise nothing will happen. The actual operation depends on the generator mode.

    @param gen_trees: list of TreeData objects representing generated trees/tokens/tagged lemmas
    @param abst_file: abstraction/delexicalization instructions file path
    @return: None
    """
    # abstss = smart_load_absts(abst_file, len(gen_trees))
    for tree in gen_trees:
        sent = lexicalizer._tree_to_sentence(tree)
        for idx, tok in enumerate(sent):
            if tok and tok.startswith('X-'):  # we would like to lexicalize
                slot = tok[2:]
                # check if we have a value to substitute; if yes, do it
                abst = lexicalizer._first_abst(absts, slot)
                if abst:
                    val = lexicalizer.get_surface_form(sent, idx, slot, abst.value)
                    tree.nodes[idx + 1] = NodeData(t_lemma=val, formeme='x')
                    sent[idx] = val  # save value to be used in LM next time
        # postprocess tokens (split multi-word nodes)
        if lexicalizer.mode == 'tokens':
            idx = 1
            while idx < len(tree):
                if ' ' in tree[idx].t_lemma:
                    value = tree[idx].t_lemma
                    tree.remove_node(idx)
                    for shift, tok in enumerate(value.split(' ')):
                        tree.create_child(0, idx + shift,
                                          NodeData(t_lemma=tok, formeme='x'))
                    idx += shift
                idx += 1


def _init_beam_search(tgen, enc_inputs):
    """Initialize beam search for the current DA (with the given encoder inputs)."""
    # initial state
    initial_state = np.zeros([1, tgen.emb_size])
    tgen._beam_search_feed_dict = {tgen.initial_state: initial_state}

    # encoder inputs
    for i in range(len(enc_inputs)):
        tgen._beam_search_feed_dict[tgen.enc_inputs[i]] = enc_inputs[i]

    empty_tree_emb = tgen.tree_embs.get_embeddings(TreeData())
    dec_inputs = np.array([[x] for x in empty_tree_emb])

    return [tgen.DecodingPath(stop_token_id=tgen.tree_embs.STOP, dec_inputs=[dec_inputs[0]])]


def rerank_beam(paths, tgen, reranker, absts, da, true_sents):
    if reranker == RERANK.TGEN:
        paths = tgen._rerank_paths(paths, da)
        return paths[0]
    elif reranker == RERANK.VANILLA:
        return paths[0]
    elif reranker in [RERANK.ORACLE, RERANK.REVERSE_ORACLE]:
        trees = [[tgen.tree_embs.ids_to_tree(ids) for ids in np.array(path.dec_inputs).transpose()] for path in paths]
        for i in range(len(trees)):
            lexicalize_beam(tgen.lexicalizer, trees[i], absts)
        toks = [t[0].to_tok_list() for t in trees]
        sents = [list(t[0] for t in tok_tuples) for tok_tuples in toks]
        acc = []
        for i, sent in enumerate(sents):
            bleu = BLEUScore()
            bleu.append(sent, true_sents)
            acc.append((bleu.score(), i))
        if reranker == RERANK.ORACLE:
            accuracy, index = max(acc)
        else:
            accuracy, index = min(acc)
        return paths[index]
    elif reranker == RERANK.RANDOM:
        idx = random.randint(0, len(paths) - 1)
        return paths[idx]
    else:
        raise ValueError("Reranker not implemented: {}", reranker)


def add_datapoint_to_training(path, tgen, da_id, label):
    hidden_state = list(path.dec_states[0].c[0])
    out_state = list(path.dec_states[0].h[0])
    pred_words = tgen.tree_embs.ids_to_strings([inp[0] for inp in path.dec_inputs])
    training_data.append((hidden_state + out_state, da_id, " ".join(pred_words), path.logprob, label))


def roll_foward_and_create_training_data(paths, tgen, da_id, beam_size, true_sents):
    new_paths = []
    # expand
    for path in paths:
        out_probs, st = tgen._beam_search_step(path.dec_inputs, path.dec_states)
        new_paths.extend(path.expand(beam_size, out_probs, st))
    # prune and create data
    pruned_paths = []
    for path in new_paths:
        pred_toks = tgen.tree_embs.ids_to_strings([inp[0] for inp in path.dec_inputs])
        pred_sent = " ".join(pred_toks[1:]).replace(" <STOP>", "")
        lab = any([(true.startswith(pred_sent)) for true in true_sents])
        add_datapoint_to_training(path, tgen, da_id, 1 if lab else 0)
        if lab and path.dec_inputs[-1] != tgen.tree_embs.STOP:
            pruned_paths.append(path)
    return pruned_paths


def rolling_beam_search(tgen, enc_inputs, beam_size, true_sents, da_id):
    _init_beam_search(tgen, enc_inputs)
    empty_tree_emb = tgen.tree_embs.get_embeddings(TreeData())
    dec_inputs = np.array([[x] for x in empty_tree_emb])

    paths = [tgen.DecodingPath(stop_token_id=tgen.tree_embs.STOP, dec_inputs=[dec_inputs[0]])]
    for step in range(len(dec_inputs)):
        paths = roll_foward_and_create_training_data(paths, tgen, da_id, beam_size, true_sents)

        if all([p.dec_inputs[-1] in [tgen.tree_embs.VOID, tgen.tree_embs.STOP] for p in paths]):
            break  # stop decoding if we have reached the end in all paths

        log_debug(("\nBEAM SEARCH STEP %d\n" % step) +
                  "\n".join([("%f\t" % p.logprob) +
                             " ".join(tgen.tree_embs.ids_to_strings([inp[0] for inp in p.dec_inputs]))
                             for p in paths]) + "\n")


def prune_from_classifier(paths, tgen, beam_size):
    global w2v_model, bc_model
    if w2v_model is None:
        w2v_model = Word2Vec.load("models/word2vec.model")
    if bc_model is None:
        bc_model = tf.keras.models.load_model('models/binary_classifiers/10_model.h5')
    features = []
    for path in paths:
        hidden_state = path.dec_states[0].c[0]
        out_state = path.dec_states[0].h[0]
        pred_words = tgen.tree_embs.ids_to_strings([inp[0] for inp in path.dec_inputs])
        embed = w2v_model.wv[remove_strange_toks(pred_words[-1])]
        prev_embed = w2v_model.wv[remove_strange_toks(pred_words[-2])]
        features.append(np.concatenate((hidden_state, out_state, embed, prev_embed)))
    acc = bc_model.predict_proba(np.array(features))
    top_hyps = np.argsort(acc,None)[-beam_size:]
    return [p for i,p in enumerate(paths) if i in top_hyps]


def _beam_search(tgen, enc_inputs, da, beam_size, absts=None, reranker=RERANK.VANILLA, true_sents=[],
                 test_pruner=False):
    """Run beam search decoding."""
    # initialize
    _init_beam_search(tgen, enc_inputs)
    empty_tree_emb = tgen.tree_embs.get_embeddings(TreeData())
    dec_inputs = np.array([[x] for x in empty_tree_emb])

    paths = [tgen.DecodingPath(stop_token_id=tgen.tree_embs.STOP, dec_inputs=[dec_inputs[0]])]

    # beam search steps
    for step in range(len(dec_inputs)):

        new_paths = []

        for path in paths:
            if path.dec_inputs[-1] not in [tgen.tree_embs.VOID, tgen.tree_embs.STOP]:
                out_probs, st = tgen._beam_search_step(path.dec_inputs, path.dec_states)
                new_paths.extend(path.expand(beam_size, out_probs, st))
            else:
                new_paths.append(path)

        if test_pruner:
            paths = prune_from_classifier(new_paths, tgen, beam_size)
        else:
            paths = sorted(new_paths,
                           key=lambda p: p.logprob,
                           reverse=True)[:beam_size]

        if all([p.dec_inputs[-1] in [tgen.tree_embs.VOID, tgen.tree_embs.STOP] for p in paths]):
            break  # stop decoding if we have reached the end in all paths

        log_debug(("\nBEAM SEARCH STEP %d\n" % step) +
                  "\n".join([("%f\t" % p.logprob) +
                             " ".join(tgen.tree_embs.ids_to_strings([inp[0] for inp in p.dec_inputs]))
                             for p in paths]) + "\n")

    best_path = rerank_beam(paths=paths,
                            tgen=tgen,
                            reranker=reranker,
                            absts=absts,
                            da=da,
                            true_sents=true_sents)
    print(" ".join(tgen.tree_embs.ids_to_strings([inp[0] for inp in best_path.dec_inputs])))
    return np.array(best_path.dec_inputs)
