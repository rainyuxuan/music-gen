import math
from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn, Tensor, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from tqdm import tqdm

from custom_types import Spectrogram
from data import SpectrogramDataset
from models.trainer import Trainer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class TransformerConfig:
    nfreqs: int
    nembed: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float


class BaseTransformer(nn.Module):
    d_model: int

    def __init__(self, nfreqs: int):
        super(BaseTransformer, self).__init__()

        self.model_type = 'Transformer'
        self.d_model = nfreqs

        # Embedding layer for the frequency bins
        self.embedding = nn.Linear(nfreqs, nfreqs)
        # nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.tgt_tok_emb = nn.Linear(nfreqs, nfreqs)

        # Positional encoding is important for maintaining the order of time frames
        self.positional_encoding = PositionalEncoding(nfreqs)

        self.transformer = nn.Transformer(d_model=nfreqs, batch_first=True)

        # Output layer
        self.output = nn.Linear(nfreqs, nfreqs)

    def forward(self, src: Spectrogram, tgt: Spectrogram) -> Spectrogram:
        src = src.permute(1, 0)
        tgt = tgt.permute(1, 0)

        src = self.embedding(src)
        src = self.positional_encoding(src)
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        output = self.transformer(src, tgt)
        output = self.output(output)
        return output

    @torch.no_grad()
    def generate(self, src: Tensor, nframes: int,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> Tensor:
        """Generate a sequence of frames from the input spectrogram"""
        result = torch.empty((src.shape[0], src.shape[1], nframes), device=device, dtype=src.dtype)

        for i in range(nframes):
            output = self.forward(src, src)
            result[:, :, i] = output[:, :, -1]
            src = torch.cat((src, output), dim=2)

        return result


class Seq2SeqTransformer(nn.Module):
    d_model: int

    def __init__(self,
                 nfreqs: int,
                 emb_size: int,):
        super(BaseTransformer, self).__init__()
        self.d_model = emb_size
        self.src_tok_emb = nn.Linear(nfreqs, emb_size)
        self.tgt_tok_emb = nn.Linear(nfreqs, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size)
        self.transformer = nn.Transformer(d_model=emb_size)
        self.generator = nn.Linear(emb_size, nfreqs)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)

    def seq2seq(self, src: Spectrogram, window_size: int) -> Spectrogram:
        """
        From a src spectrogram batch, slide the window over the time axis. For each window of src spectrogram sequence,
        generate the output spectrogram sequence of length output_nframes. The sliding window has stride of output_nframes.
        :param src: input spectrogram batch, shape (batch_size, nfreqs, nframes)
        :param window_size: number of frames in the sliding window (length of the sequence for the encoder and decoder)
        :return: output spectrogram batch, shape (batch_size, nfreqs, nframes)
        """
        # TODO: Implement this method
        num_output_frames = src.shape[2] - window_size + 1
        output = torch.empty((src.shape[0], src.shape[1], num_output_frames), device=src.device, dtype=src.dtype)

        for i in range(num_output_frames):
            src_window = src[:, :, i:i+window_size]
            tgt_window = src[:, :, i + 1:i + 1 + window_size]
            output_window = self.forward(src_window, tgt_window)
            output[:, :, i:i+1] = output_window[:, :, -1:]

        return output

    @torch.no_grad()
    def generate(self, src: Tensor, nframes: int,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> Tensor:
        """Generate a sequence of frames from the input spectrogram"""
        result = torch.empty((src.shape[0], src.shape[1], nframes), device=device, dtype=src.dtype)
        for i in range(nframes):
            if i == 0:
                tgt = src[:, :, i:i+1]
            else:
                tgt = result[:, :, i-1:i]
            output = self.forward(src, tgt)
            result[:, :, i:i+1] = output[:, :, -1:]
        return result


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer model, copied from PyTorch tutorial
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    return src_mask, tgt_mask


def train_epoch(model, optimizer, dl):
    model.train()
    losses = 0

    for src, tgt in dl:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(dl))


if __name__ == "__main__":
    dataset = SpectrogramDataset(data_dir="data", seq_first=True)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)

    nfreqs = dataset[0][0].shape[0]

    transformer = BaseTransformer(nfreqs=nfreqs, emb_size=nfreqs)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)




