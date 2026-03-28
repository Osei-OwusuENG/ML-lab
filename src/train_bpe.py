import os
from collections import defaultdict
import numpy as np
import re
import json


def get_corpus(text):

    words = re.findall(r"\w+|[^\w\s]", text.lower())

    return [list(word) + ['</w>'] for word in words]

def get_pairs(word):

    pairs = [pair for pair in zip(word, word[1:])]

    return pairs

def count_pairs(corpus):
    
    pairs_freq = defaultdict(int)

    for word in corpus:
        pairs = get_pairs(word)
        # counts = np.unique_count(pairs)
        # pairs_freq[pair] = counts[pair].count()
        for pair in pairs:
            pairs_freq[pair] += 1

    return pairs_freq

def merge_pairs(pair, corpus):

    new_corpus = []

    bigram = ''.join(pair)

    for word in corpus:
        new_word = []

        i = 0

        while i < len(word):

            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                new_word.append(bigram)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_corpus.append(new_word)

    return new_corpus

def train_bpe(text, num_merges=10):

    corpus = get_corpus(text)
    merges = []

    for _ in range(num_merges):

        pairs_freq = count_pairs(corpus)

        if not pairs_freq:
            break

        best_pairs = max(pairs_freq, key=pairs_freq.get)
        merges.append(best_pairs)

        corpus = merge_pairs(best_pairs, corpus)

    return merges, corpus

def save_bpe(merges, v=0.2, path="./data/processed/vocab.bpe"):
    seen = set()
    unique_merges = []

    for a, b in merges:
        if (a, b) not in seen:
            seen.add((a, b))
            unique_merges.append((a, b))

    with open(path, "w", encoding="utf-8") as f:
        f.write("#version: 0.1\n")
        for a, b in unique_merges:
            f.write(f"{a} {b}\n")


def load_bpe(filepath="./data/processed/vocab.bpe"):
    merges = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            a, b = line.strip().split()
            merges.append((a, b))
    return merges

