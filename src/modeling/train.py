from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import TrainingConfig
from src.modeling.architecture import build_person_detector
from src.modeling.coco_dataset import CocoPersonDataset
from src.modeling.voc_dataset import PascalVocPersonDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def build_dataset(
    annotations: str,
    images: str,
    transforms,
):
    annotation_path = Path(annotations)
    if annotation_path.suffix.lower() == ".json":
        return CocoPersonDataset(annotations, images, transforms)

    if annotation_path.suffix.lower() == ".txt":
        dataset_root = annotation_path.parent
        xml_root = dataset_root / "Annotations"
        image_root = dataset_root / "JPEGImages"
        if not xml_root.exists():
            raise FileNotFoundError(f"No existe la carpeta XML: {xml_root}")
        if not image_root.exists():
            raise FileNotFoundError(f"No existe la carpeta de imagenes: {image_root}")
        return PascalVocPersonDataset(
            str(xml_root),
            str(image_root),
            split_file=str(annotation_path),
            transforms=transforms,
        )

    if annotation_path.is_dir():
        image_root = Path(images)
        return PascalVocPersonDataset(
            str(annotation_path),
            str(image_root),
            transforms=transforms,
        )

    raise ValueError(
        "Las anotaciones deben ser un JSON COCO, un archivo .txt de particion "
        "VOC o una carpeta con XML."
    )


def train(
    annotations: str,
    images: str,
    output: str,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> Path:
    config = TrainingConfig()
    if epochs is not None:
        config = TrainingConfig(epochs=epochs)
    if batch_size is not None:
        config = TrainingConfig(
            batch_size=batch_size,
            epochs=config.epochs,
        )

    transforms = T.Compose([T.PILToTensor(), T.ConvertImageDtype(torch.float32)])
    dataset = build_dataset(annotations, images, transforms)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_person_detector(config.num_classes).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for images_batch, targets in progress:
            images_batch = [image.to(device) for image in images_batch]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            losses = model(images_batch, targets)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            progress.set_postfix(loss=float(loss.item()))

        print(f"Epoch {epoch + 1} loss: {epoch_loss / max(len(loader), 1):.4f}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Modelo guardado en {output_path}")
    return output_path

