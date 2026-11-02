"""Exercise train/evaluate/predict CLIs on generated fixtures; no downloads or real data."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from image_captioning.config import Config


def run(*arguments, env):
    result = subprocess.run(
        [sys.executable, "-m", *arguments], check=True, env=env, text=True, capture_output=True
    )
    print(result.stdout.strip())
    return result


def main():
    with tempfile.TemporaryDirectory(prefix="caption-smoke-") as directory:
        root = Path(directory)
        images = root / "images"
        images.mkdir()
        rows = ["image_name|comment_number|comment"]
        for index in range(8):
            name = f"{index}.jpg"
            Image.new("RGB", (70, 80), (index * 30, 100, 150)).save(images / name)
            rows.extend([f"{name}|0|a dog runs .", f"{name}|1|a dog jumps ."])
        annotations = root / "results.csv"
        annotations.write_text("\n".join(rows) + "\n", encoding="utf-8")
        config = Config(
            captions_file=str(annotations),
            image_dir=str(images),
            output_dir=str(root / "artifacts"),
            embed_size=16,
            num_heads=2,
            num_layers=1,
            resize=64,
            crop_size=64,
            max_length=12,
            num_workers=0,
            batch_size=2,
            epochs=1,
            pretrained_encoder=False,
            word2vec_min_count=1,
        )
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        run("image_captioning.train", "--config", str(config_path), "--device", "cpu", env=env)
        checkpoint = str(root / "artifacts/best.pt")
        output = root / "metrics.json"
        run(
            "image_captioning.evaluate",
            "--checkpoint",
            checkpoint,
            "--device",
            "cpu",
            "--output",
            str(output),
            env=env,
        )
        metrics = json.loads(output.read_text())
        assert metrics["split"] == "test" and metrics["tokens"] > 0
        run(
            "image_captioning.predict",
            "--checkpoint",
            checkpoint,
            "--image",
            str(images / "0.jpg"),
            "--max-tokens",
            "5",
            "--device",
            "cpu",
            env=env,
        )
        # A changed annotation file must not silently redefine the test set.
        annotations.write_text(annotations.read_text() + "\n", encoding="utf-8")
        rejected = subprocess.run(
            [
                sys.executable,
                "-m",
                "image_captioning.evaluate",
                "--checkpoint",
                checkpoint,
                "--device",
                "cpu",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0 and "differs from training" in rejected.stderr
        print(
            "Smoke test passed: train, checkpoint reload, evaluation, prediction, and split guard."
        )


if __name__ == "__main__":
    main()
