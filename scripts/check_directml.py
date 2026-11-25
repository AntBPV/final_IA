import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    try:
        import torch_directml
    except ImportError:
        print("torch-directml no esta instalado.")
        return 1

    device = torch_directml.device()
    print(f"Dispositivo DirectML: {device}")

    left = torch.tensor([1.0], device=device)
    right = torch.tensor([2.0], device=device)
    print(f"Prueba tensor: {(left + right).item()}")

    from src.modeling.architecture import build_person_detector

    model = build_person_detector(2).eval().to(device)
    image = torch.zeros((3, 320, 320), dtype=torch.float32, device=device)
    with torch.no_grad():
        result = model([image])[0]
    print("Claves:", sorted(result.keys()))
    print("Cajas:", len(result["boxes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
