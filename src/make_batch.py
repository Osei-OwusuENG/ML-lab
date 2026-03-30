import torch


def make_batch(padded, device="cpu"):
    batch = torch.tensor(padded, dtype=torch.long, device=device)
    return batch