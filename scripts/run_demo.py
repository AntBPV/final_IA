import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.demo import PersonDemo


def main():
    parser = argparse.ArgumentParser(description="Ejecutar la demostracion")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="models/person_detector.torchscript")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "dml", "auto"],
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--infer-every", type=int, default=2)
    args = parser.parse_args()
    demo = PersonDemo(
        args.model,
        args.threshold,
        device_name=args.device,
        max_width=args.width,
        infer_every=args.infer_every,
    )
    demo.run(args.source)


if __name__ == "__main__":
    main()

