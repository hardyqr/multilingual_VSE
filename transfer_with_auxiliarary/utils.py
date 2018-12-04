#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import numpy as np
import collections
import sys
import pickle
import torch.nn as nn
import torch


# import from torch-goodies
from embeddings import FastTextVocab
from embeddings import FastTextEmbeddings



class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab_from_dict(fname):
    """
    fname - path to a .txt dictionary
    """
    counter_left = [] 
    counter_right = []
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    for i,line in enumerate(fin):
        w_left,w_right = line.split(" ")
        counter_left.append(w_left)
        counter_right.append(w_right)
    counter_left = set(counter_left)
    counter_right = set(counter_right)

    vocab_left = Vocabulary()
    vocab_right = Vocabulary()
    for word in counter_left.items():
        vocab_left.add_word(word)
    for word in counter_right.items():
        vocab_right.add_word(word)
    
    return vocab_left, vocab_right

class subword_embed():
    def __init__(self,vocab_path,ft_vocab_path,path_to_binary_file,path_to_computed_embed,lang='de'):
        self.vanilla_vocab = None
        self.ft_vocab = None
        self.ft_embeddings = None
        self.embedding_sum = None
        #path_to_binary_file='/mnt/storage01/fangyu/fasttext_embeddings/wiki.de.bin'
        if path_to_computed_embed is None:
            self.ft_embeddings = FastTextEmbeddings(wv_file=path_to_binary_file)
        if ft_vocab_path is None:
            self.vanilla_vocab = pickle.load(open(vocab_path,'rb'))
            # construct FastTextVocab
            
            print ('[merging old vocab with FastTextVocab...]')
            self.ft_vocab = FastTextVocab(
                      include_word=True,
                      min_subword=4, max_subword=6,
                      word_min_freq=1, subword_min_freq=3)
            self.ft_vocab.create_index_mapping(
                         self.ft_embeddings, vocab=self.vanilla_vocab.word2idx )
        
            # save ft_vocab
            with open('./vocab/ft_vocab_'+lang+'_min4_max_6_1_3_.pkl','wb') as f:
                pickle.dump(self.ft_vocab,f,pickle.HIGHEST_PROTOCOL)
        else:
            self.ft_vocab = pickle.load(open(ft_vocab_path,'rb'))
        
        self.embedding_sum = nn.EmbeddingBag(len(self.ft_vocab), 300, mode='sum')
        if path_to_computed_embed is None:
            #self.ft_embeddings = FastTextEmbeddings(wv_file=path_to_binary_file)
            self.ft_embeddings.init_weights(
                    self.embedding_sum.weight.data, 
                    self.ft_vocab.vocab())
            torch.save(self.embedding_sum.state_dict(),'./embed/vocab_'+lang+'_4_6_1_3_embedding_bag_.pth')
        else:
            # if weights provided
            self.embedding_sum.load_state_dict(torch.load(path_to_computed_embed))

    def compute_embed(self,ws):
        """
        Arguments:
            ws - a list of strings (tokens)
        """
        print ('computing embed for: {}'.format(ws))
        input_,offset_ = self.ft_vocab.numericalize([ws])[0]
        print ('indexes: {}, offsets: {}'.format(input_, offset_))
        word_vecs = self.embedding_sum(torch.LongTensor(input_), torch.LongTensor(offset_))
        return word_vecs
        
def load_vectors(fname, maxload=200000, norm=True, center=False, verbose=True):
    """
    Returns:
        words - all the words (string).
        x - a list of vectors. size (n,d).
    """
    if verbose: # n. 啰嗦
        print("Loading vectors from %s" % fname)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split()) # first line of .vec is "vec_num word_dim"
    print (n,d)
    if maxload > 0:
        n = min(n, maxload)
    print (n,d)
    x = np.zeros([n, d])
    words = []
    for i, line in enumerate(fin):
        if i >= n:
            break
        tokens = line.rstrip().split(' ')
        words.append(tokens[0])
        v = np.array(tokens[1:], dtype=float)
        x[i, :] = v
    if norm:
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if center:
        x -= x.mean(axis=0)[np.newaxis, :]
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if verbose:
        print("%d word vectors loaded" % (len(words)))
    return words, x


def idx(words):
    w2i = {}
    for i, w in enumerate(words):
        if w not in w2i:
            w2i[w] = i
    return w2i


def save_vectors(fname, x, words):
    n, d = x.shape
    fout = io.open(fname, 'w', encoding='utf-8')
    fout.write(u"%d %d\n" % (n, d))
    for i in range(n):
        fout.write(words[i] + " " + " ".join(map(lambda a: "%.4f" % a, x[i, :])) + "\n")
    fout.close()


def save_matrix(fname, x):
    n, d = x.shape
    fout = io.open(fname, 'w', encoding='utf-8')
    fout.write(u"%d %d\n" % (n, d))
    for i in range(n):
        fout.write(" ".join(map(lambda a: "%.4f" % a, x[i, :])) + "\n")
    fout.close()


def load_matrix(fname):
    fin = io.open(fname,encoding='utf-8',newline='\n',errors='ignore')
    n,d = map(int,fin.readline().split())
    print (n,d)
    m = []
    for i,line in enumerate(fin):
        m.append(np.array(line.rstrip().split(' '),'float'))
    m = np.array(m)
    print (m.shape)
    return m



def procrustes(X_src, Y_tgt):
    U, s, V = np.linalg.svd(np.dot(Y_tgt.T, X_src))
    return np.dot(U, V)


def select_vectors_from_pairs(x_src, y_tgt, pairs):
    n = len(pairs)
    d = x_src.shape[1]
    x = np.zeros([n, d])
    y = np.zeros([n, d])
    for k, ij in enumerate(pairs):
        i, j = ij
        x[k, :] = x_src[i, :]
        y[k, :] = y_tgt[j, :]
    return x, y


def load_lexicon(filename, words_src, words_tgt, verbose=True):
    """
    lexicon, n.词典
    """
    f = io.open(filename, 'r', encoding='utf-8') # open dict
    lexicon = collections.defaultdict(set) # declare lexicon
    idx_src , idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        word_src, word_tgt = line.split()[0:2]
        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
        vocab.add(word_src)
    if verbose:
        coverage = len(lexicon) / float(len(vocab))
        print("Coverage of source vocab: %.4f" % (coverage))
    return lexicon, float(len(vocab))


def load_pairs(filename, idx_src, idx_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    pairs = []
    tot = 0
    for line in f:
        a, b = line.rstrip().split(' ')[0:2]
        tot += 1
        if a in idx_src and b in idx_tgt:
            pairs.append((idx_src[a], idx_tgt[b]))
    if verbose:
        coverage = (1.0 * len(pairs)) / tot
        print("Found pairs for training: %d - Total pairs in file: %d - Coverage of pairs: %.4f" % (len(pairs), tot, coverage))
    return pairs


def compute_nn_accuracy(x_src, x_tgt, lexicon, bsz=100, lexicon_size=-1):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())
    acc = 0.0
    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8
    for i in range(0, len(idx_src), bsz):
        e = min(i + bsz, len(idx_src))
        scores = np.dot(x_tgt, x_src[idx_src[i:e]].T)
        pred = scores.argmax(axis=0)
        for j in range(i, e):
            if pred[j - i] in lexicon[idx_src[j]]:
                acc += 1.0
    return acc / lexicon_size


def compute_csls_accuracy(x_src, x_tgt, lexicon, lexicon_size=-1, k=10, bsz=1024):
    if lexicon_size < 0:
        lexicon_size = len(lexicon)
    idx_src = list(lexicon.keys())

    x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8
    x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8

    sr = x_src[list(idx_src)]
    sc = np.dot(sr, x_tgt.T)
    similarities = 2 * sc
    sc2 = np.zeros(x_tgt.shape[0])
    for i in range(0, x_tgt.shape[0], bsz):
        j = min(i + bsz, x_tgt.shape[0])
        sc_batch = np.dot(x_tgt[i:j, :], x_src.T)
        dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
        sc2[i:j] = np.mean(dotprod, axis=1)
    similarities -= sc2[np.newaxis, :]

    nn = np.argmax(similarities, axis=1).tolist()
    correct = 0.0
    for k in range(0, len(lexicon)):
        if nn[k] in lexicon[idx_src[k]]:
            correct += 1.0
    return correct / lexicon_size
