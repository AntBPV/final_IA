import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def build_person_detector(num_classes: int = 2) -> FasterRCNN:
    """Construye Faster R-CNN desde cero, sin cargar pesos preentrenados."""
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=None,
        trainable_layers=5,
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=None,
        box_score_thresh=0.5,
    )
    return model

