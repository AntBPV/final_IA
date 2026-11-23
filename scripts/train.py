import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.export import export_torchscript
from src.modeling.train import train

def main():
    parser = argparse.ArgumentParser(description="Entrenar el detector desde cero")
    parser.add_argument(
        "--annotations",
        default="data/raw/train/train.txt",
        help="JSON COCO, train.txt VOC o carpeta de XML",
    )
    parser.add_argument(
        "--images",
        default="data/raw/train/JPEGImages",
        help="Carpeta de imagenes para COCO o XML",
    )
    parser.add_argument("--checkpoint", default="models/person_detector.pth")
    parser.add_argument("--output", default="models/person_detector.torchscript")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "dml"],
        help="Dispositivo: auto, cpu, cuda o dml",
    )
    parser.add_argument(
        "--validation-annotations",
        default="data/raw/val/val.txt",
        help="Val .txt, JSON COCO, XML o carpeta",
    )
    parser.add_argument(
        "--validation-images",
        default="data/raw/val/JPEGImages",
        help="Carpeta de imagenes de validacion",
    )
    args = parser.parse_args()

    checkpoint = train(
        args.annotations,
        args.images,
        args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_name=args.device,
        validation_annotations=args.validation_annotations,
        validation_images=args.validation_images,
    )
    export_torchscript(str(checkpoint), args.output)


if __name__ == "__main__":
    main()

