import math
from dataclasses import dataclass

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder

from custom_types import Spectrogram
from models import device


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
        self.src_tok_emb = nn.Linear(nfreqs, nfreqs)
        # self.tgt_tok_emb = nn.Linear(nfreqs, nfreqs)

        # Positional encoding is important for maintaining the order of time frames
        self.positional_encoding = PositionalEncoding(nfreqs)

        self.transformer = nn.Transformer(d_model=nfreqs, batch_first=True)

        # Output layer
        self.output = nn.Linear(nfreqs, nfreqs)

    def forward(self, src: Spectrogram, tgt: Spectrogram = None) -> Spectrogram:
        src = self.src_tok_emb(src)  # Embed source spectrogram

        src = self.positional_encoding(src)

        output = self.transformer(src)
        output = self.output(output)
        return output

    @torch.no_grad()
    def generate(self, src: Tensor, nframes: int) -> Tensor:
        """Generate a sequence of frames from the input spectrogram"""
        result = torch.empty((src.shape[0], src.shape[1], nframes), device=device, dtype=src.dtype)

        for i in range(nframes):
            output = self.forward(src)
            result[:, :, i] = output[:, :, -1]
            src = torch.cat((src, output), dim=2)

        return result


class ModularizedTransformer(BaseTransformer):
    nhead: int

    def __init__(self, nfreqs: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, dim_feedforward=2048,
                 dropout=0.1):
        super(ModularizedTransformer, self).__init__(nfreqs)

        self.model_type = 'Transformer'
        self.nhead = nhead

        self.pos_encoder = PositionalEncoding(nfreqs, dropout)

        encoder_layers = TransformerEncoderLayer(nfreqs, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        decoder_layers = TransformerDecoderLayer(nfreqs, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

        self.embedding = nn.Embedding(nfreqs, nfreqs)
        # self.linear = nn.Linear(nfreqs, nfreqs)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # self.linear.bias.data.zero_()
        # self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Spectrogram, src_mask: Tensor = None) -> Spectrogram:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        code = self.transformer_encoder(src, src_mask)
        output = self.transformer_decoder(code)
        # output = self.linear(output)
        return output


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
