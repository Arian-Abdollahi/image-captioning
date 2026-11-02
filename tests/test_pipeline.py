"""Semantic checks for the refactor; no Flickr30k data or pretrained downloads."""

import math
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from image_captioning.checkpoint import load_model, save_checkpoint
from image_captioning.config import Config
from image_captioning.data import (
    FlickrDataset,
    ImageRecord,
    collate_captions,
    make_transforms,
    read_annotations,
    split_records,
)
from image_captioning.embeddings import build_embedding_weights
from image_captioning.engine import run_epoch
from image_captioning.model import DecoderTransformer, ImageCaptioningModel, PositionalEncoding
from image_captioning.predict import generate_caption
from image_captioning.vocabulary import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX, Vocabulary

torch.set_num_threads(2)


class DataTests(unittest.TestCase):
    def test_captions_stay_separate_and_splits_do_not_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text(
                "image_name|comment_number|comment\n"
                "a.jpg|0|A DOG runs.\na.jpg|1|A dog jumps.\n"
                "b.jpg|0|A cat.\nc.jpg|0|A bird.\n",
                encoding="utf-8",
            )
            records = read_annotations(path)
        self.assertEqual(records[0].captions, ("a dog runs.", "a dog jumps."))
        splits = split_records(records, seed=5)
        self.assertEqual(splits, split_records(list(reversed(records)), seed=5))
        ids = [{record.image_id for record in items} for items in splits.values()]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        train_texts = [text for record in splits["train"] for text in record.captions]
        vocabulary = Vocabulary.build(train_texts)
        self.assertEqual(vocabulary.encode("unseenword")[1], UNK_IDX)
        self.assertNotIn("[", vocabulary.word_to_index)

    def test_truncation_preserves_eos_and_padding_is_batch_first(self):
        vocabulary = Vocabulary.build(["one two three four five"])
        long = torch.tensor(vocabulary.encode("one two three four five", max_length=4))
        short = torch.tensor(vocabulary.encode("one", max_length=4))
        images, captions = collate_captions(
            [(torch.zeros(3, 8, 8), long), (torch.zeros(3, 8, 8), short)]
        )
        self.assertEqual(long.tolist()[-1], EOS_IDX)
        self.assertEqual(tuple(images.shape), (2, 3, 8, 8))
        self.assertEqual(tuple(captions.shape), (2, 4))
        self.assertEqual(captions[1, -1].item(), PAD_IDX)

    def test_evaluation_images_are_repeatable_and_missing_files_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            Image.new("RGB", (50, 60), "green").save(Path(directory) / "a.jpg")
            vocabulary = Vocabulary.build(["a dog"])
            dataset = FlickrDataset(
                [ImageRecord("a.jpg", ("a dog",))], directory, vocabulary, make_transforms(40, 32)
            )
            self.assertTrue(torch.equal(dataset[0][0], dataset[0][0]))
            Path(directory, "a.jpg").unlink()
            with self.assertRaises(FileNotFoundError):
                dataset[0]

    def test_bad_annotations_fail_with_a_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text("image|index|caption\nbroken\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row 2"):
                read_annotations(path)

    def test_word2vec_accepts_words_and_zeroes_padding(self):
        config = Config(embed_size=8, num_heads=2, word2vec_min_count=1)
        texts = ["a dog runs", "a cat runs", "a dog jumps"]
        vocabulary = Vocabulary.build(texts)
        first = build_embedding_weights(texts, vocabulary, config)
        second = build_embedding_weights(texts, vocabulary, config)
        self.assertEqual(tuple(first.shape), (len(vocabulary), 8))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(first[PAD_IDX].eq(0).all())
        self.assertTrue(torch.isfinite(first).all())


class ModelTests(unittest.TestCase):
    def test_decoder_cannot_see_future_tokens(self):
        decoder = DecoderTransformer(8, 12, num_heads=2, num_layers=1, dropout=0).eval()
        features = torch.randn(1, 8)
        with torch.no_grad():
            first = decoder(features, torch.tensor([[SOS_IDX, 4, 5]]))
            second = decoder(features, torch.tensor([[SOS_IDX, 4, 9]]))
        torch.testing.assert_close(first[:, :2], second[:, :2])

    def test_padding_does_not_change_metrics(self):
        class UniformModel(nn.Module):
            def forward(self, images, captions):
                return torch.zeros(*captions.shape, 6)

        model = UniformModel()
        images = torch.zeros(1, 3, 8, 8)
        original = run_epoch(
            model,
            [(images, torch.tensor([[SOS_IDX, 4, EOS_IDX]]))],
            torch.device("cpu"),
            progress=False,
        )
        padded = run_epoch(
            model,
            [(images, torch.tensor([[SOS_IDX, 4, EOS_IDX, 0, 0]]))],
            torch.device("cpu"),
            progress=False,
        )
        self.assertEqual(original["tokens"], 2)
        self.assertAlmostEqual(original["loss"], math.log(6), places=6)
        self.assertAlmostEqual(original["loss"], padded["loss"], places=6)
        self.assertAlmostEqual(padded["perplexity"], 6, places=5)

    def test_training_checkpoint_roundtrip_and_autoregressive_inference(self):
        config = Config(
            embed_size=8,
            num_heads=2,
            num_layers=1,
            max_length=8,
            dropout=0,
            pretrained_encoder=False,
            batch_size=2,
        )
        vocabulary = Vocabulary.build(["a dog runs"])
        model = ImageCaptioningModel(config, len(vocabulary))
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        images = torch.randn(2, 3, 64, 64)
        captions = torch.tensor([vocabulary.encode("a dog"), vocabulary.encode("a dog")])
        before = model.encoder.fc.weight.detach().clone()
        running_mean = model.encoder.resnet[1].running_mean.clone()
        metrics = run_epoch(
            model, [(images, captions)], torch.device("cpu"), optimizer, progress=False
        )
        self.assertFalse(torch.equal(before, model.encoder.fc.weight))
        self.assertTrue(torch.equal(running_mean, model.encoder.resnet[1].running_mean))
        self.assertTrue(all(p.grad is None for p in model.encoder.resnet.parameters()))
        model.eval()
        with torch.no_grad():
            expected = model(images, captions[:, :-1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                config,
                vocabulary,
                {"train": ["a.jpg"], "validation": ["b.jpg"], "test": ["c.jpg"]},
                "fixture",
                1,
                metrics,
            )
            restored, restored_vocabulary, restored_config, payload = load_model(
                path,
                torch.device("cpu"),
            )
            with torch.no_grad():
                actual = restored(images, captions[:, :-1])
            torch.testing.assert_close(actual, expected)
            self.assertEqual(restored_config.max_length, 8)
            self.assertEqual(payload["epoch"], 1)
            caption = generate_caption(restored, images[0], restored_vocabulary, 3, 8)
            self.assertIsInstance(caption, str)
            self.assertNotIn("<PAD>", caption)
            self.assertNotIn("<SOS>", caption)

    def test_odd_embedding_width_and_length_guard(self):
        positional_encoding = PositionalEncoding(9, max_length=4)
        self.assertEqual(tuple(positional_encoding(torch.zeros(4, 2, 9)).shape), (4, 2, 9))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            positional_encoding(torch.zeros(5, 2, 9))

    def test_invalid_attention_configuration_fails_early(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            Config(embed_size=300, num_heads=8).validate()


if __name__ == "__main__":
    unittest.main()
