from pathlib import Path

import torch

from src.config import TrainingConfig
from src.modeling.architecture import build_person_detector


def export_torchscript(checkpoint: str, output: str) -> Path:
    device = torch.device("cpu")
    model = build_person_detector(TrainingConfig.num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    traced = torch.jit.script(model)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    print(f"Modelo exportado en {output_path}")
    return output_path

