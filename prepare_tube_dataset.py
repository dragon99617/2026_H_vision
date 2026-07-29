#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_label(path: Path, require_ball) -> int:
    count = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError("invalid class/field count: %s:%d" % (path, line_number))
        values = [float(value) for value in fields[1:]]
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("normalized label outside 0..1: %s:%d" % (path, line_number))
        if values[2] <= 0.0 or values[3] <= 0.0:
            raise ValueError("non-positive label extent: %s:%d" % (path, line_number))
        count += 1
    if require_ball is True and count != 1:
        raise ValueError(
            "positive half-pipe image must have exactly one reviewed ball label: %s"
            % path
        )
    if require_ball is False and count:
        raise ValueError("negative image must have an empty label: %s" % path)
    return count


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge reviewed continuous half-pipe segments with legacy YOLO data"
    )
    parser.add_argument(
        "--raw", type=Path, default=PROJECT_DIR / "tube_data/raw"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_DIR / "dataset_tube"
    )
    parser.add_argument("--minimum-images", type=int, default=300)
    parser.add_argument("--minimum-negatives", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(
            "output already exists; move it aside before rebuilding: %s"
            % args.output
        )
    segment_files = sorted(args.raw.glob("*/segment.json"))
    if not segment_files:
        raise RuntimeError("no collected segment manifests under %s" % args.raw)
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in segment_files
    ]
    collected = sum(int(item["captured_images"]) for item in manifests)
    negatives = sum(
        int(item["captured_images"])
        for item in manifests
        if item["mode"] == "negative"
    )
    if collected < args.minimum_images:
        raise RuntimeError(
            "need at least %d actual half-pipe images; found %d"
            % (args.minimum_images, collected)
        )
    if negatives < args.minimum_negatives:
        raise RuntimeError(
            "need at least %d negative images; found %d"
            % (args.minimum_negatives, negatives)
        )
    splits_present = {item["split"] for item in manifests}
    if splits_present != {"train", "val", "test"}:
        raise RuntimeError(
            "continuous segments must cover train, val and test; found %s"
            % sorted(splits_present)
        )

    report = {
        "legacy_dataset": str(PROJECT_DIR / "dataset"),
        "raw_tube_dataset": str(args.raw),
        "segments": [],
        "splits": {},
    }
    for split in ("train", "val", "test"):
        image_out = args.output / "images" / split
        label_out = args.output / "labels" / split
        image_out.mkdir(parents=True)
        label_out.mkdir(parents=True)
        split_images = 0
        split_boxes = 0
        legacy_images = sorted((PROJECT_DIR / "dataset/images" / split).glob("*"))
        for image in legacy_images:
            label = PROJECT_DIR / "dataset/labels" / split / (image.stem + ".txt")
            if not label.is_file():
                raise FileNotFoundError(label)
            destination_stem = "legacy_" + image.stem
            link_or_copy(image, image_out / (destination_stem + image.suffix.lower()))
            link_or_copy(label, label_out / (destination_stem + ".txt"))
            split_images += 1
            split_boxes += validate_label(label, require_ball=None)

        for manifest_path, manifest in zip(segment_files, manifests):
            if manifest["split"] != split:
                continue
            segment_dir = manifest_path.parent
            mode = manifest["mode"]
            segment_report = {
                "segment": manifest["segment"],
                "split": split,
                "mode": mode,
                "images": 0,
            }
            for record in manifest["records"]:
                image = segment_dir / record["image"]
                label = segment_dir / record["label"]
                if not image.is_file() or not label.is_file():
                    raise FileNotFoundError("missing collected image/label in %s" % segment_dir)
                if sha256(image) != record["image_sha256"]:
                    raise ValueError("collected image hash changed: %s" % image)
                boxes = validate_label(label, require_ball=(mode == "positive"))
                stem = "tube_%s_%s" % (manifest["segment"], image.stem)
                link_or_copy(image, image_out / (stem + image.suffix.lower()))
                link_or_copy(label, label_out / (stem + ".txt"))
                split_images += 1
                split_boxes += boxes
                segment_report["images"] += 1
            report["segments"].append(segment_report)
        report["splits"][split] = {
            "images": split_images,
            "boxes": split_boxes,
        }
    data_yaml = (
        "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: ball\n"
    ) % args.output.resolve()
    (args.output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (args.output / "split_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
