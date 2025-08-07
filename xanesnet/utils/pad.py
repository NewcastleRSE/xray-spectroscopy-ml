import torch
from torch.nn.utils.rnn import pad_sequence

def pad(sequence):
    return pad_sequence(sequence, batch_first=True)


def pad_all(sequences):
    padded_sequences = tuple([pad(seq) for seq in sequences])
    return padded_sequences