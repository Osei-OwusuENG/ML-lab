from pathlib import Path

from tokenizer.BPEtokenizer import BPETokenizer


BASE_DIR = Path(__file__).resolve().parents[1]
CORPUS_PATH = BASE_DIR / "data" / "raw" / "corpus.txt"
VOCAB_PATH = BASE_DIR / "data" / "processed" / "Lvocab.json"
MERGES_PATH = BASE_DIR / "data" / "processed" / "Lmerges.json"


def main():

    # corpus = CORPUS_PATH.read_text(encoding='utf-8').splitlines()

    tokenizer = BPETokenizer()
    # tokenizer.learn_bpe(corpus, num_merges=20000, min_freq=2)
    # tokenizer.build_vocab()
    # tokenizer.save(VOCAB_PATH, MERGES_PATH)

    tokenizer.load(VOCAB_PATH, MERGES_PATH)

    sample_text = '''"The quuick browne fox jumps over the lazy dog.""unseenwordxyz""Sphinx of black quartz, judged my vow!""Pack my box with five dozena liquor jugs.""don't""i'm here""you're right""it's working""hello!""hello, world.""what are you doing?""yes: i agree.""The five boxing wizards jump quickly."'''
    
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print(f"Sample text: {sample_text}")
    print(f'len of encoded: {len(encoded)}')
    print(f"Encoded sample: {encoded}")
    print(f'len of decoded: {len(decoded)}')
    print(f"Decoded sample: {decoded}")


if __name__ == "__main__":
    main()
