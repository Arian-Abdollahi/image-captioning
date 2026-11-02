"""Word2Vec initialization using tokenized training captions only."""

import hashlib

import numpy as np
import torch

from .vocabulary import PAD_IDX, SPECIAL_TOKENS, tokenize


def _stable_hash(word):
    return int.from_bytes(hashlib.md5(word.encode("utf-8")).digest()[:4], "little")


def build_embedding_weights(captions, vocabulary, config):
    from gensim.models import Word2Vec

    sentences = [tokenize(caption) for caption in captions]
    model = Word2Vec(
        sentences=sentences,
        vector_size=config.embed_size,
        window=config.word2vec_window,
        min_count=config.word2vec_min_count,
        epochs=config.word2vec_epochs,
        workers=config.word2vec_workers,
        seed=config.seed,
        hashfxn=_stable_hash,
    )
    rng = np.random.default_rng(config.seed)
    weights = rng.normal(0, 0.6, (len(vocabulary), config.embed_size)).astype(np.float32)
    for word, index in vocabulary.word_to_index.items():
        if word not in SPECIAL_TOKENS and word in model.wv:
            weights[index] = model.wv[word]
    weights[PAD_IDX] = 0
    return torch.from_numpy(weights)
