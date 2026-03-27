import os
import re
from collections import defaultdict


# CREATE TOKENIZER CLASS
class BPETokenizer():
    def __init__(self, merges):
        self.merges = merges
        self.bpe_ranks = [{pair: i for pair, i in enumerate(self.merges)}]

        self.encoder = {} # STORE TOKEN -> ID
        self.decoder = {} # STORE ID -> TOKEN

    def apply_bpe(self, word):
        
        word = list(word) + '</w>'
        
        while True:
            new_word = []

            pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]

            if not pairs:
                break

            rank_pair = [(pair, self.bpe_ranks.get(pair, float('inf'))) for pair in pairs]
            best_pair = min(rank_pair, key=lambda x: x(1))[0]

            if best_pair not in rank_pair:
                break


            i = 0

            while i < len(word):

                if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                    new_word.append(best_pair)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word

        return word
    
    def BuildVacab(self, corpus):

        token = set()

        for sentence in corpus:
            words = [re.findall(r"\w+|[^w\s]"), sentence.lower()]
        
            for word in words:
                token.update(self.apply_bpe(word, self.bpe_ranks))

            self.encoder = {tok:id for id, tok in enumerate(sorted(token))}
            self.decoder = {id:tok for tok, id in self.encoder.item()}

    # ENCODE
    def encode(self, text):
        token = []
        words = [re.findall("\w+|[^w\s]"), text.lower()]

        for word in words:
            token.extend(self.apply_bpe(word))
        
        ids = [self.encoder[tok] for tok in token]

        return token, ids

    # DECODE
    def decode(self, ids):

        token = [self.decoder[id] for id in ids]

        text = ''.join(token)
        text = text.replace('</w>', ' ')

        return text.strip()