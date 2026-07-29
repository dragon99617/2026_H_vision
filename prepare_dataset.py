#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_DIR.parent / "data/labels"
DEFAULT_OUTPUT = PROJECT_DIR / "dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Sample:
    stem: str
    image: Path
    label: Path
    captured: datetime
    object_count: int

    @property
    def positive(self) -> bool:
        return self.object_count > 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_capture_time(stem: str) -> datetime:
    fields = stem.split("_")
    if len(fields) < 4:
        raise ValueError("unexpected filename: %s" % stem)
    return datetime.strptime(fields[2] + fields[3], "%Y%m%d%H%M%S")


def validate_label(path: Path) -> int:
    count = 0
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("%s:%d must have 5 fields" % (path, line_number))
        class_id = int(fields[0])
        x, y, width, height = map(float, fields[1:])
        if class_id != 0:
            raise ValueError("%s:%d class must be 0" % (path, line_number))
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError("%s:%d contains a non-finite value" % (path, line_number))
        if not (
            0.0 <= x <= 1.0
            and 0.0 <= y <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise ValueError("%s:%d contains an invalid box" % (path, line_number))
        if (
            x - width * 0.5 < -1e-5
            or x + width * 0.5 > 1.0 + 1e-5
            or y - height * 0.5 < -1e-5
            or y + height * 0.5 > 1.0 + 1e-5
        ):
            raise ValueError("%s:%d box leaves the image" % (path, line_number))
        count += 1
    return count


def discover_samples(source: Path) -> List[Sample]:
    images = {
        path.stem: path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    labels = {path.stem: path for path in source.glob("*.txt")}
    if set(images) != set(labels):
        raise ValueError(
            "unpaired data: images_only=%s labels_only=%s"
            % (sorted(set(images) - set(labels)), sorted(set(labels) - set(images)))
        )
    samples = []
    image_hashes = set()
    for stem in sorted(images):
        image = images[stem]
        matrix = cv2.imread(str(image))
        if matrix is None:
            raise ValueError("cannot read image: %s" % image)
        if matrix.shape[:2] != (800, 1280):
            raise ValueError(
                "%s is %dx%d, expected 1280x800"
                % (image, matrix.shape[1], matrix.shape[0])
            )
        image_hash = sha256(image)
        if image_hash in image_hashes:
            raise ValueError("duplicate image content found: %s" % image)
        image_hashes.add(image_hash)
        samples.append(
            Sample(
                stem=stem,
                image=image,
                label=labels[stem],
                captured=parse_capture_time(stem),
                object_count=validate_label(labels[stem]),
            )
        )
    return samples


def contiguous_sessions(
    samples: Sequence[Sample],
    maximum_gap_seconds: float = 10.0,
) -> List[List[Sample]]:
    sessions: List[List[Sample]] = []
    for sample in sorted(samples, key=lambda item: item.captured):
        if (
            not sessions
            or (sample.captured - sessions[-1][-1].captured).total_seconds()
            > maximum_gap_seconds
        ):
            sessions.append([sample])
        else:
            sessions[-1].append(sample)
    return sessions


def planned_split(samples: Sequence[Sample]) -> Dict[str, List[Sample]]:
    positives = [sample for sample in samples if sample.positive]
    negatives = [sample for sample in samples if not sample.positive]
    positive_sessions = contiguous_sessions(positives)
    negative_sessions = contiguous_sessions(negatives)
    positive_sizes = [len(group) for group in positive_sessions]
    negative_sizes = [len(group) for group in negative_sessions]
    if positive_sizes != [31, 19, 30] or negative_sizes != [9]:
        raise ValueError(
            "dataset changed; expected positive sessions [31, 19, 30] and "
            "negative sessions [9], got %s and %s"
            % (positive_sizes, negative_sizes)
        )
    held_out_positive = positive_sessions[2]
    negative = negative_sessions[0]
    return {
        "train": positive_sessions[0]
        + positive_sessions[1]
        + negative[:5],
        "val": held_out_positive[:15] + negative[5:7],
        "test": held_out_positive[15:] + negative[7:],
    }


def copy_sample(sample: Sample, split: str, output: Path) -> None:
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.image, image_dir / sample.image.name)
    shutil.copy2(sample.label, label_dir / sample.label.name)


def motion_blur(image: np.ndarray, size: int, horizontal: bool) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    if horizontal:
        kernel[size // 2, :] = 1.0 / size
    else:
        kernel[:, size // 2] = 1.0 / size
    return cv2.filter2D(image, -1, kernel)


def add_shadow(image: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]
    x1 = rng.randint(-width // 2, width)
    x2 = rng.randint(0, width + width // 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(
        [[x1, 0], [x1 + width // 3, 0], [x2 + width // 3, height], [x2, height]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [polygon], 255)
    factor = rng.uniform(0.72, 0.88)
    result = image.astype(np.float32)
    result[mask > 0] *= factor
    return np.clip(result, 0, 255).astype(np.uint8)


def create_augmented_sample(
    sample: Sample,
    output: Path,
    rng: random.Random,
) -> None:
    image = cv2.imread(str(sample.image))
    alpha = rng.uniform(0.82, 1.18)
    beta = rng.uniform(-18.0, 18.0)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    image = add_shadow(image, rng)
    image = motion_blur(image, rng.choice((3, 5)), rng.random() < 0.5)
    name = "aug_motion_" + sample.image.name
    image_dir = output / "augmented/images"
    label_dir = output / "augmented/labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_dir / name), image):
        raise OSError("failed to write augmented image")
    shutil.copy2(sample.label, label_dir / ("aug_motion_" + sample.label.name))


def manifest_entry(sample: Sample, split: str) -> dict:
    return {
        "stem": sample.stem,
        "split": split,
        "captured": sample.captured.isoformat(),
        "positive": sample.positive,
        "object_count": sample.object_count,
        "image_sha256": sha256(sample.image),
        "label_sha256": sha256(sample.label),
        "source_image": str(sample.image.resolve()),
        "source_label": str(sample.label.resolve()),
    }


def prepare(source: Path, output: Path) -> dict:
    samples = discover_samples(source)
    split = planned_split(samples)
    expected = {
        "train": (55, 50, 5),
        "val": (17, 15, 2),
        "test": (17, 15, 2),
    }
    for name, values in split.items():
        actual = (
            len(values),
            sum(item.positive for item in values),
            sum(not item.positive for item in values),
        )
        if actual != expected[name]:
            raise AssertionError("%s split is %s, expected %s" % (name, actual, expected[name]))

    if output.exists():
        if output.resolve() == source.resolve() or output.resolve() == PROJECT_DIR.resolve():
            raise ValueError("refusing to replace unsafe output path")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    entries = []
    for name, values in split.items():
        for sample in values:
            copy_sample(sample, name, output)
            entries.append(manifest_entry(sample, name))

    rng = random.Random(2026)
    for sample in split["train"]:
        create_augmented_sample(sample, output, rng)

    manifest = {
        "version": 1,
        "source": str(source.resolve()),
        "seed": 2026,
        "raw_images": len(samples),
        "objects": sum(item.object_count for item in samples),
        "classes": {"0": "ball"},
        "split_counts": {
            name: {
                "images": len(values),
                "positive": sum(item.positive for item in values),
                "negative": sum(not item.positive for item in values),
                "objects": sum(item.object_count for item in values),
            }
            for name, values in split.items()
        },
        "augmented_train_images": len(split["train"]),
        "samples": entries,
        "note": (
            "All nine negative images come from one contiguous capture session; "
            "negative-background generalization requires live validation."
        ),
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    data_yaml = (
        "path: %s\n"
        "train:\n"
        "  - images/train\n"
        "  - augmented/images\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: ball\n"
        % output.resolve()
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and split the ball dataset")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = prepare(args.source.resolve(), args.output.resolve())
    print(json.dumps(manifest["split_counts"], ensure_ascii=False, indent=2))
    print("augmented_train_images=%d" % manifest["augmented_train_images"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
