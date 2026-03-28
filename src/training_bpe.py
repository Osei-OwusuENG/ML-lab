from pathlib import Path

from BPEtokenizer import BPETokenizer


BASE_DIR = Path(__file__).resolve().parents[1]
CORPUS_PATH = BASE_DIR / "data" / "raw" / "corpus.txt"
VOCAB_PATH = BASE_DIR / "data" / "processed" / "vocab.json"
MERGES_PATH = BASE_DIR / "data" / "processed" / "merges.json"


def main():
    text = CORPUS_PATH.read_text(encoding="utf-8")

    tokenizer = BPETokenizer()
    tokenizer.learn_bpe(text, num_merges=30000, min_freq=2)
    vocab = tokenizer.build_vocab()
    tokenizer.save(VOCAB_PATH, MERGES_PATH)

    sample_text = "Artificial intelligence is transforming industries."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print(f"Corpus path: {CORPUS_PATH}")
    print(f"Learned merges: {len(tokenizer.merges)}")
    print(f"Vocabulary size: {len(tokenizer.encoder)}")
    print(f"Sample text: {sample_text}")
    print(f"Encoded sample: {encoded}")
    print(f"Decoded sample: {decoded}")
    print(f"Vocabulary: {vocab}")


if __name__ == "__main__":
    main()
