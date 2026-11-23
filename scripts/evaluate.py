import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.train import build_dataset


def box_iou(box, ground_truth):
    x1 = max(box[0], ground_truth[0])
    y1 = max(box[1], ground_truth[1])
    x2 = min(box[2], ground_truth[2])
    y2 = min(box[3], ground_truth[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_truth = (
        max(0.0, ground_truth[2] - ground_truth[0])
        * max(0.0, ground_truth[3] - ground_truth[1])
    )
    union = area_box + area_truth - intersection
    return intersection / union if union > 0 else 0.0


def average_precision_at_iou(predictions, ground_truth, iou_threshold=0.5):
    detections = sorted(predictions, key=lambda item: item[0], reverse=True)
    matched = [set() for _ in ground_truth]
    true_positives = []
    false_positives = []

    for _, image_index, box in detections:
        truth_boxes = ground_truth[image_index]
        best_iou = 0.0
        best_truth = -1
        for truth_index, truth_box in enumerate(truth_boxes):
            if truth_index in matched[image_index]:
                continue
            iou = box_iou(box, truth_box)
            if iou > best_iou:
                best_iou = iou
                best_truth = truth_index
        if best_iou >= iou_threshold and best_truth >= 0:
            matched[image_index].add(best_truth)
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    total_truth = sum(len(boxes) for boxes in ground_truth)
    if total_truth == 0:
        return 0.0

    cumulative_tp = 0
    cumulative_fp = 0
    precision = []
    recall = []
    for true_positive, false_positive in zip(true_positives, false_positives):
        cumulative_tp += true_positive
        cumulative_fp += false_positive
        precision.append(cumulative_tp / (cumulative_tp + cumulative_fp))
        recall.append(cumulative_tp / total_truth)

    return _integral_precision_recall(precision, recall)


def _integral_precision_recall(precision, recall):
    recall_values = [0.0] + recall + [1.0]
    precision_values = [0.0] + precision + [0.0]
    for index in range(len(precision_values) - 2, -1, -1):
        precision_values[index] = max(
            precision_values[index],
            precision_values[index + 1],
        )
    average_precision = 0.0
    for index in range(1, len(recall_values)):
        average_precision += (
            recall_values[index] - recall_values[index - 1]
        ) * precision_values[index]
    return average_precision


def main():
    parser = argparse.ArgumentParser(
        description="Evaluar detecciones sobre el conjunto especificado"
    )
    parser.add_argument(
        "--partition",
        choices=["val", "test"],
        help="Usa una particion conocida; test requiere una evaluacion final explicita",
    )
    parser.add_argument(
        "--annotations",
        default="data/raw/val/val.txt",
    )
    parser.add_argument(
        "--images",
        default="data/raw/val/JPEGImages",
    )
    parser.add_argument(
        "--model",
        default="models/person_detector.torchscript",
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.partition:
        if args.partition == "test":
            args.annotations = "data/raw/Test/test.txt"
            args.images = "data/raw/Test/JPEGImages"
        else:
            args.annotations = "data/raw/val/val.txt"
            args.images = "data/raw/val/JPEGImages"

    transforms = T.Compose(
        [T.PILToTensor(), T.ConvertImageDtype(torch.float32)]
    )
    dataset = build_dataset(args.annotations, args.images, transforms)
    model = torch.jit.load(args.model, map_location="cpu")
    model.eval()

    predictions = []
    ground_truth = []
    for image_index in range(len(dataset)):
        image, target = dataset[image_index]
        ground_truth.append(
            target["boxes"].cpu().tolist()
        )
        with torch.no_grad():
            result = model(image)
        boxes = result["boxes"].cpu().tolist()
        labels = result["labels"].cpu().tolist()
        scores = result["scores"].cpu().tolist()
        for box, label, score in zip(boxes, labels, scores):
            if label == 1 and score >= args.score_threshold:
                predictions.append((score, image_index, box))

    ap50 = average_precision_at_iou(predictions, ground_truth, 0.5)
    print(f"Imagenes evaluadas: {len(dataset)}")
    print(f"Cajas reales: {sum(len(boxes) for boxes in ground_truth)}")
    print(f"Detecciones: {len(predictions)}")
    print(f"mAP@0.50: {ap50:.4f}")


if __name__ == "__main__":
    main()
