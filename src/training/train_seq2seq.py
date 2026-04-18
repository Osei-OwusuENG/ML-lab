import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import ast
from src.tokenizer.BPEtokenizer import BPETokenizer
from src.models import lstmencoderModel, decoderModel, seq2seqModel
import pyfiglet
from rich.console import Console


BASE_DIR = Path(__file__).resolve().parents[2]
SEQ_PATH = BASE_DIR / "data" / "raw" / "seqdataset.txt"
VOCAB_PATH = BASE_DIR / "data" / "processed" / "Lvocab.json"
MERGES_PATH = BASE_DIR / "data" / "processed" / "Lmerges.json"

tokenizer = BPETokenizer()

console = Console()

tokenizer.load(VOCAB_PATH, MERGES_PATH)

corpus = SEQ_PATH.read_text()
corpus = ast.literal_eval(corpus.split("=", 1)[1].strip())

def tokenize_corpus(corpus, tokenizer):
    input_seqs = [pair[0] for pair in corpus]
    output_seqs = [pair[1] for pair in corpus]

    vocab_size = len(tokenizer.encoder)
    # max_id = max(tokenizer.encoder.values()) + 1
    # print(f"Tokenizer vocabulary size: {vocab_size}, max token ID: {max_id}")

    tokens_input = [tokenizer.encode(seq, add_special_tokens=True) for seq in input_seqs]
    tokens_output = [tokenizer.encode(seq, add_special_tokens=True) for seq in output_seqs]

    tokens_input_padded, input_mask = tokenizer.pad_sequence_with_mask(tokens_input, padding_side="right")
    tokens_input_padded = torch.tensor(tokens_input_padded)
    input_mask = torch.tensor(input_mask)
    tokens_output_padded, output_mask = tokenizer.pad_sequence_with_mask(tokens_output, padding_side="right")
    tokens_output_padded = torch.tensor(tokens_output_padded)
    output_mask = torch.tensor(output_mask)


    # print('input seqs:', input_seqs[:5])
    # print('output seqs:', output_seqs[:5])

    # print('encoded input seqs:', tokens_input[:5])
    # print('encoded output seqs:', tokens_output[:5])

    # print('padded input seqs:', tokens_input_padded[:5])
    # print('input mask:', input_mask[:5])

    # print('padded output seqs:', tokens_output_padded[:5])
    # print('output mask:', output_mask[:5])

    # decoded_input = [tokenizer.decode(tokens_ids) for tokens_ids in tokens_input]
    # decoded_output = [tokenizer.decode(tokens_ids) for tokens_ids in tokens_output]

    # print('decoded input seqs:', decoded_input[:5])
    # print('decoded output seqs:', decoded_output[:5])

    return tokens_input_padded, input_mask, tokens_output_padded, output_mask, vocab_size


tokens_input_padded, input_mask, tokens_output_padded, output_mask, vocab_size = tokenize_corpus(corpus, tokenizer)

def generate_test_tokens(src, trg, md, tokenizer, max_len = 30):
    md.eval()
    hidden, cell = md.encoder(src)
    input = torch.tensor([tokenizer.encoder[tokenizer.start_token]], device=src.device)  # (1,)

    outputs = []
    for _ in range(max_len):
        input = input.unsqueeze(0)  # (1, 1)

        output, hidden, cell = md.decoder(input, hidden, cell)

        pred_token = output.argmax(2).item()

        if pred_token == tokenizer.encoder[tokenizer.end_token]:
            break

        input = torch.tensor([pred_token], device=src.device)
        outputs.append(pred_token)

    return outputs


def train(tokens_input_padded, input_mask, tokens_output_padded, output_mask, vocab_size) -> None:
    # create optimizer, loss function, and dataloader here
    model = seq2seqModel(
        lstmencoderModel(vocab_size, embedding_dim=64, hidden_dim=128),
        decoderModel(vocab_size, embedding_dim=64, hidden_dim=128),
        vocab_size
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.encoder[tokenizer.pad_token])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(2000):
        optimizer.zero_grad()

        outputs = model(tokens_input_padded, tokens_output_padded)

        # reshape outputs and targets for loss calculation
        outputs = outputs[:, 1:, :].reshape(-1, vocab_size)  # exclude <sos> token
        targets = tokens_output_padded[:, 1:].reshape(-1)  # exclude <sos> token

        loss = criterion(outputs, targets)
        loss.backward()
        Grad_Norm =torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            non_pad = targets != tokenizer.encoder[tokenizer.pad_token]
            correct = (preds[non_pad] == targets[non_pad]).sum().item()
            total = non_pad.sum().item()
            accuracy = correct / total if total > 0 else 0



        if epoch % 10 == 0:
            print()
            num = pyfiglet.figlet_format(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Grad Norm: {Grad_Norm:.4f}", font="block")
            console.print(num, style="bold cyan")
            print()

            for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                test_src = tokens_input_padded[idx:idx+1]
                test_trg = tokens_output_padded[idx]

                test_output_tokens = generate_test_tokens(test_src, test_trg, max_len=30, md=model, tokenizer=tokenizer)

                Test = pyfiglet.figlet_format(f"Test Input: {tokenizer.decode(test_src[0].tolist())}", font="block")
                console.print(Test, style="bold green")

                Target = pyfiglet.figlet_format(f"Test target: {tokenizer.decode(test_trg.tolist())}", font="block")
                console.print(Target, style="bold blue")

                Output = pyfiglet.figlet_format(f"Test output_tokens: {tokenizer.decode(test_output_tokens)}", font="block")
                console.print(Output, style="bold yellow")



if __name__ == "__main__":
    train(tokens_input_padded, input_mask, tokens_output_padded, output_mask, vocab_size)
    print("Training completed.")