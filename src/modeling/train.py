from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import TrainingConfig
from src.modeling.architecture import build_person_detector
from src.modeling.coco_dataset import CocoPersonDataset
from src.modeling.voc_dataset import PascalVocPersonDataset
from src.device import get_device


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
    device_name: str = "auto",
    validation_annotations: str | None = None,
    validation_images: str | None = None,
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

    validation_loader = None
    if validation_annotations and validation_images:
        validation_dataset = build_dataset(
            validation_annotations,
            validation_images,
            transforms,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

    device = get_device(device_name)
    print(f"Dispositivo de entrenamiento: {device}")
    model = build_person_detector(config.num_classes).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    model.train()
    best_validation_loss = float("inf")
    best_state = None
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

        validation_loss = None
        if validation_loader is not None:
            model.train()
            validation_loss_sum = 0.0
            with torch.no_grad():
                for validation_images_batch, validation_targets in validation_loader:
                    validation_images_batch = [
                        image.to(device) for image in validation_images_batch
                    ]
                    validation_targets = [
                        {
                            key: value.to(device)
                            for key, value in target.items()
                        }
                        for target in validation_targets
                    ]
                    losses = model(validation_images_batch, validation_targets)
                    validation_loss_sum += float(
                        sum(losses.values()).item()
                    )
            validation_loss = validation_loss_sum / max(len(validation_loader), 1)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        message = (
            f"Epoch {epoch + 1} loss: "
            f"{epoch_loss / max(len(loader), 1):.4f}"
        )
        if validation_loss is not None:
            message += f" val_loss: {validation_loss:.4f}"
        print(message)

    if best_state is not None:
        model.load_state_dict(best_state)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Modelo guardado en {output_path}")
    return output_path

