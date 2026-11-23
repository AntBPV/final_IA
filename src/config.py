from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    image_size: int = 800
    batch_size: int = 2
    epochs: int = 20
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    num_classes: int = 2
    score_threshold: float = 0.5

