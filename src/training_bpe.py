from BPEtokenizer import BPETokenizer
from train_bpe import train_bpe
from train_bpe import save_bpe


text = open('./data/raw/corpus.txt').read()

merges, corpus = train_bpe(text, num_merges=10000)

tokenizer = BPETokenizer(merges=merges)
token = tokenizer.BuildVacab(corpus=text)

tokenizer.save_vacab()

save_bpe(merges, v=0.1)
print('token:', token)
print(merges)

