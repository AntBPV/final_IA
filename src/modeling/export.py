from pathlib import Path

import torch

from src.config import TrainingConfig
from src.modeling.architecture import build_person_detector


class DetectionExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image):
        detections = self.model([image])[0]
        return detections


def export_torchscript(checkpoint: str, output: str) -> Path:
    device = torch.device("cpu")
    model = build_person_detector(TrainingConfig.num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    wrapper = DetectionExportWrapper(model).eval()
    example = torch.zeros((3, 640, 640), dtype=torch.float32)
    traced = torch.jit.trace(wrapper, example, strict=False)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    print(f"Modelo exportado en {output_path}")
    return output_path

