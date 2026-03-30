from pathlib import Path

from BPEtokenizer import BPETokenizer


BASE_DIR = Path(__file__).resolve().parents[1]
CORPUS_PATH = BASE_DIR / "data" / "raw" / "corpus.txt"
VOCAB_PATH = BASE_DIR / "data" / "processed" / "Evocab.json"
MERGES_PATH = BASE_DIR / "data" / "processed" / "Emerges.json"


def main():

    tokenizer = BPETokenizer()
    
    tokenizer.load(VOCAB_PATH, MERGES_PATH)

    sample_text = '''"The quuick browne fox jumps over the lazy dog.""unseenwordxyz""Sphinx of black quartz, judged my vow!""Pack my box with five dozena liquor jugs.""The five boxing wizards jump quickly."'''
    
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print(f"Sample text: {sample_text}")
    print(f'len of encoded: {len(encoded)}')
    print(f"Encoded sample: {encoded}")
    print(f'len of decoded: {len(decoded)}')
    print(f"Decoded sample: {decoded}")


if __name__ == "__main__":
    main()
