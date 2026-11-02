"""Frozen ResNet-50 image encoder and autoregressive Transformer caption decoder."""

import math

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from .vocabulary import PAD_IDX


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, pretrained=True):
        super().__init__()
        # V1 preserves the notebook's legacy pretrained=True initialization.
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet50(weights=weights)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet.requires_grad_(False)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.resnet.eval()

    def train(self, mode=True):
        super().train(mode)
        # Freeze running statistics as well as parameters in the pretrained backbone.
        self.resnet.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images).flatten(1)
        return self.bn(self.fc(features))


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_length=100):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        pe = torch.zeros(max_length, 1, embed_size)
        pe[:, 0, 0::2] = torch.sin(position * divisor)
        pe[:, 0, 1::2] = torch.cos(position * divisor[: embed_size // 2])
        self.register_buffer("pe", pe)

    def forward(self, inputs):
        if inputs.size(0) > self.pe.size(0):
            raise ValueError("Caption exceeds the configured positional encoding length")
        return inputs + self.pe[: inputs.size(0)]


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        embed_size,
        vocab_size,
        num_heads=10,
        num_layers=3,
        embedding_weights=None,
        dropout=0.1,
        max_length=100,
    ):
        super().__init__()
        if embed_size % num_heads:
            raise ValueError("embed_size must be divisible by num_heads")
        if embedding_weights is None:
            self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        else:
            if tuple(embedding_weights.shape) != (vocab_size, embed_size):
                raise ValueError("Embedding weights must match vocabulary size and embed_size")
            self.embed = nn.Embedding.from_pretrained(
                embedding_weights,
                freeze=False,
                padding_idx=PAD_IDX,
            )
        self.pos_encoder = PositionalEncoding(embed_size, max_length)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        # [B,T] -> [T,B,E]; a single image vector is the cross-attention memory.
        target = self.pos_encoder(self.embed(captions).transpose(0, 1))
        length = captions.size(1)
        causal_mask = torch.ones(length, length, dtype=torch.bool, device=captions.device).triu(1)
        output = self.transformer_decoder(
            target,
            features.unsqueeze(0),
            tgt_mask=causal_mask,
            tgt_key_padding_mask=captions.eq(PAD_IDX),
        )
        return self.fc_out(output).transpose(0, 1)


class ImageCaptioningModel(nn.Module):
    def __init__(self, config, vocab_size, embedding_weights=None, *, pretrained=None):
        super().__init__()
        config.validate()
        if pretrained is None:
            pretrained = config.pretrained_encoder
        self.encoder = EncoderCNN(config.embed_size, pretrained)
        self.decoder = DecoderTransformer(
            config.embed_size,
            vocab_size,
            config.num_heads,
            config.num_layers,
            embedding_weights,
            config.dropout,
            config.max_length,
        )

    def forward(self, images, captions):
        return self.decoder(self.encoder(images), captions)
