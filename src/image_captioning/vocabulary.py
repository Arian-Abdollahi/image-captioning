"""spaCy word tokenization and a JSON-serializable training vocabulary."""

from collections import Counter
from functools import lru_cache

SPECIAL_TOKENS = ("<PAD>", "<SOS>", "<EOS>", "<UNK>")
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = range(4)


@lru_cache(maxsize=1)
def _tokenizer():
    import spacy

    # Only the tokenizer is required; no en_core_web_sm download or tagger is needed.
    return spacy.blank("en").tokenizer


def tokenize(text):
    return [token.text.lower() for token in _tokenizer()(text) if not token.is_space]


class Vocabulary:
    def __init__(self, word_to_index):
        self.word_to_index = dict(word_to_index)
        if any(self.word_to_index.get(token) != i for i, token in enumerate(SPECIAL_TOKENS)):
            raise ValueError("Vocabulary must contain the four special tokens at indices 0..3")
        values = list(self.word_to_index.values())
        if any(type(i) is not int for i in values) or sorted(values) != list(range(len(values))):
            raise ValueError("Vocabulary indices must be unique, contiguous integers")
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}

    @classmethod
    def build(cls, captions, min_frequency=1):
        counter = Counter(token for text in captions for token in tokenize(text))
        mapping = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        # Stable tie-breaking makes the mapping independent of row order.
        for word, frequency in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if frequency >= min_frequency and word not in mapping:
                mapping[word] = len(mapping)
        return cls(mapping)

    def __len__(self):
        return len(self.word_to_index)

    def encode(self, text, max_length=100):
        if max_length < 3:
            raise ValueError("max_length must be >= 3")
        words = tokenize(text)[: max_length - 2]
        return [SOS_IDX] + [self.word_to_index.get(w, UNK_IDX) for w in words] + [EOS_IDX]

    def decode(self, indices):
        words = []
        for index in indices:
            index = int(index)
            if index == EOS_IDX:
                break
            if index not in (PAD_IDX, SOS_IDX):
                words.append(self.index_to_word.get(index, "<UNK>"))
        return " ".join(words)
