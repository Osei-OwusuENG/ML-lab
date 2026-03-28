import json
import re
from collections import defaultdict


class BPETokenizer:
    WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(self, lowercase=True, unk_token="[UNK]", end_of_word="</w>"):
        self.lowercase = lowercase
        self.unk_token = unk_token
        self.end_of_word = end_of_word

        self.merges = []
        self.bpe_ranks = {}
        self.encoder = {}
        self.decoder = {}

        self._alphabet = set()
        self._word_freq = {}
        self._cache = {}

    def _normalize_text(self, text):
        return text.lower() if self.lowercase else text

    def _iter_texts(self, corpus):
        if isinstance(corpus, str):
            yield corpus
            return

        for text in corpus:
            if text is None:
                continue
            yield str(text)

    def _pre_tokenize(self, text):
        return self.WORD_PATTERN.findall(self._normalize_text(text))

    def _word_to_symbols(self, word):
        return tuple(list(word) + [self.end_of_word])

    def _merge_word(self, word, pair):
        merged = []
        merged_token = "".join(pair)
        index = 0

        while index < len(word):
            if index < len(word) - 1 and (word[index], word[index + 1]) == pair:
                merged.append(merged_token)
                index += 2
                continue

            merged.append(word[index])
            index += 1

        return tuple(merged)

    def _get_pair_stats(self, word_freq):
        pairs = defaultdict(int)
        for word, freq in word_freq.items():
            for index in range(len(word) - 1):
                pairs[(word[index], word[index + 1])] += freq
        return pairs

    def _count_words(self, corpus):
        word_freq = defaultdict(int)
        alphabet = set()

        for text in self._iter_texts(corpus):
            for word in self._pre_tokenize(text):
                symbols = self._word_to_symbols(word)
                word_freq[symbols] += 1
                alphabet.update(symbols[:-1])

        return word_freq, alphabet

    def learn_bpe(self, corpus, num_merges=10000, min_freq=2):
        word_freq, alphabet = self._count_words(corpus)
        if not word_freq:
            raise ValueError("Cannot learn BPE from an empty corpus.")

        self.merges = []
        self.bpe_ranks = {}
        self.encoder = {}
        self.decoder = {}
        self._cache = {}
        self._alphabet = alphabet

        for merge_idx in range(num_merges):
            pairs = self._get_pair_stats(word_freq)
            if not pairs:
                break

            best_pair, best_freq = max(pairs.items(), key=lambda item: item[1])
            if best_freq < min_freq:
                break

            self.merges.append(best_pair)
            self.bpe_ranks[best_pair] = merge_idx

            merged_word_freq = defaultdict(int)
            for word, freq in word_freq.items():
                merged_word_freq[self._merge_word(word, best_pair)] += freq
            word_freq = merged_word_freq

            if (merge_idx + 1) % 1000 == 0:
                print(f"Learned {merge_idx + 1} merges...")

        self._word_freq = dict(word_freq)
        print(f"Successfully learned {len(self.merges)} merges")
        return self.merges

    def build_vocab(self):
        token_set = {self.unk_token, self.end_of_word}
        token_set.update(self._alphabet)
        token_set.update("".join(pair) for pair in self.merges)

        for word in self._word_freq:
            token_set.update(word)

        sorted_tokens = sorted(token_set, key=lambda token: (len(token), token))
        self.encoder = {token: index for index, token in enumerate(sorted_tokens)}
        self.decoder = {index: token for token, index in self.encoder.items()}

        print(f"Vocabulary built with {len(self.encoder)} tokens")
        return self.encoder

    def apply_bpe(self, word):
        normalized_word = self._normalize_text(word)
        if not normalized_word:
            return []

        cached = self._cache.get(normalized_word)
        if cached is not None:
            return list(cached)

        symbols = self._word_to_symbols(normalized_word)
        while len(symbols) > 1:
            available_pairs = [
                pair
                for pair in zip(symbols, symbols[1:])
                if pair in self.bpe_ranks
            ]
            if not available_pairs:
                break

            best_pair = min(available_pairs, key=lambda pair: self.bpe_ranks[pair])
            symbols = self._merge_word(symbols, best_pair)

        self._cache[normalized_word] = symbols
        return list(symbols)

    def encode(self, text):
        if not self.encoder:
            raise ValueError("Build or load the vocabulary before encoding text.")

        unknown_id = self.encoder[self.unk_token]
        encoded = []

        for word in self._pre_tokenize(text):
            for token in self.apply_bpe(word):
                encoded.append(self.encoder.get(token, unknown_id))

        return encoded

    def decode(self, token_ids):
        if not self.decoder:
            raise ValueError("Build or load the vocabulary before decoding token ids.")

        words = []
        current_word = ""

        for token_id in token_ids:
            token = self.decoder.get(token_id, self.unk_token)

            if token == self.unk_token:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(token)
                continue

            if token == self.end_of_word:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                continue

            if token.endswith(self.end_of_word):
                current_word += token[: -len(self.end_of_word)]
                words.append(current_word)
                current_word = ""
                continue

            current_word += token

        if current_word:
            words.append(current_word)

        text = " ".join(words)
        text = re.sub(r"\s+([,.;:!?%)\]\}])", r"\1", text)
        text = re.sub(r"([(\[\{])\s+", r"\1", text)
        return text

    def save(self, vocab_path="vocab.json", merges_path="merges.json"):
        with open(vocab_path, "w", encoding="utf-8") as vocab_file:
            json.dump(self.encoder, vocab_file, indent=2, ensure_ascii=False)

        merges_list = [list(pair) for pair in self.merges]
        with open(merges_path, "w", encoding="utf-8") as merges_file:
            json.dump(merges_list, merges_file, indent=2, ensure_ascii=False)

        print(f"Saved vocab to {vocab_path}")
        print(f"Saved merges to {merges_path}")

    def load(self, vocab_path="vocab.json", merges_path="merges.json"):
        with open(vocab_path, "r", encoding="utf-8") as vocab_file:
            self.encoder = json.load(vocab_file)
        self.decoder = {index: token for token, index in self.encoder.items()}

        with open(merges_path, "r", encoding="utf-8") as merges_file:
            merges_list = json.load(merges_file)

        self.merges = [tuple(pair) for pair in merges_list]
        self.bpe_ranks = {pair: index for index, pair in enumerate(self.merges)}
        self._cache = {}

        print(f"Loaded {len(self.encoder)} tokens and {len(self.merges)} merges")
        return self
