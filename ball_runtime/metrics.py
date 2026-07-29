from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .types import Detection


@dataclass(frozen=True)
class ImagePredictions:
    image: Path
    ground_truth: Tuple[Detection, ...]
    predictions: Tuple[Detection, ...]


def load_yolo_ground_truth(image: Path, label: Path) -> Tuple[Detection, ...]:
    import cv2

    matrix = cv2.imread(str(image))
    if matrix is None:
        raise ValueError("cannot read image: %s" % image)
    height, width = matrix.shape[:2]
    boxes = []
    for raw in label.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if not fields:
            continue
        class_id = int(fields[0])
        center_x, center_y, box_width, box_height = map(float, fields[1:])
        boxes.append(
            Detection(
                x1=(center_x - box_width * 0.5) * width,
                y1=(center_y - box_height * 0.5) * height,
                x2=(center_x + box_width * 0.5) * width,
                y2=(center_y + box_height * 0.5) * height,
                confidence=1.0,
                class_id=class_id,
            )
        )
    return tuple(boxes)


def iou(left: Detection, right: Detection) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def match_image(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    threshold: float,
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    unmatched = set(range(len(ground_truth)))
    true_positives = 0
    false_positives = 0
    selected = sorted(
        (item for item in predictions if item.confidence >= threshold),
        key=lambda item: item.confidence,
        reverse=True,
    )
    for prediction in selected:
        candidates = [
            (iou(prediction, ground_truth[index]), index)
            for index in unmatched
        ]
        overlap, index = max(candidates, default=(0.0, -1))
        if overlap >= iou_threshold:
            unmatched.remove(index)
            true_positives += 1
        else:
            false_positives += 1
    return true_positives, false_positives, len(unmatched)


def score(
    records: Sequence[ImagePredictions],
    threshold: float,
    iou_threshold: float = 0.5,
) -> dict:
    true_positives = false_positives = false_negatives = 0
    for record in records:
        tp, fp, fn = match_image(
            record.ground_truth,
            record.predictions,
            threshold,
            iou_threshold,
        )
        true_positives += tp
        false_positives += fp
        false_negatives += fn
    precision = (
        true_positives / float(true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / float(true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def select_threshold(
    records: Sequence[ImagePredictions],
    target_recall: float = 0.97,
) -> dict:
    scores = sorted(
        {
            float(prediction.confidence)
            for record in records
            for prediction in record.predictions
        }
        | {0.001, 0.999}
    )
    evaluated = [score(records, threshold) for threshold in scores]
    recall_candidates = [
        item for item in evaluated if item["recall"] >= target_recall
    ]
    if recall_candidates:
        selected = max(
            recall_candidates,
            key=lambda item: (item["precision"], item["threshold"]),
        )
        rule = "max_precision_at_recall_gte_%.3f" % target_recall
    else:
        selected = max(
            evaluated,
            key=lambda item: (item["f1"], item["recall"], item["precision"]),
        )
        rule = "max_f1"
    return dict(selected, selection_rule=rule)


def average_precision_50(records: Sequence[ImagePredictions]) -> float:
    total_ground_truth = sum(len(record.ground_truth) for record in records)
    if total_ground_truth == 0:
        return 0.0
    ranked = []
    for record in records:
        unmatched = set(range(len(record.ground_truth)))
        predictions = sorted(
            record.predictions,
            key=lambda item: item.confidence,
            reverse=True,
        )
        for prediction in predictions:
            candidates = [
                (iou(prediction, record.ground_truth[index]), index)
                for index in unmatched
            ]
            overlap, index = max(candidates, default=(0.0, -1))
            is_true = overlap >= 0.5
            if is_true:
                unmatched.remove(index)
            ranked.append((prediction.confidence, 1 if is_true else 0))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return 0.0
    truth = np.asarray([item[1] for item in ranked], dtype=np.float64)
    cumulative_tp = np.cumsum(truth)
    cumulative_fp = np.cumsum(1.0 - truth)
    recall = cumulative_tp / float(total_ground_truth)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    grid = np.linspace(0.0, 1.0, 101)
    interpolated = np.asarray(
        [
            np.max(precision[recall >= point])
            if np.any(recall >= point)
            else 0.0
            for point in grid
        ],
        dtype=np.float64,
    )
    return float(np.mean(interpolated))


def summarize(
    records: Sequence[ImagePredictions],
    threshold: float,
) -> dict:
    result = score(records, threshold)
    result["map50"] = average_precision_50(records)
    result["images"] = len(records)
    result["ground_truth_objects"] = sum(
        len(record.ground_truth) for record in records
    )
    result["raw_predictions"] = sum(
        len(record.predictions) for record in records
    )
    return result
