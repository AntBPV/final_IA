import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class CocoPersonDataset(Dataset):
    """Dataset COCO reducido a la clase persona."""

    def __init__(self, annotations_path: str, images_root: str, transforms=None):
        with open(annotations_path, "r", encoding="utf-8") as handle:
            coco = json.load(handle)

        self.images_root = Path(images_root)
        self.transforms = transforms
        person_categories = [
            category["id"]
            for category in coco.get("categories", [])
            if category.get("name", "").lower() in {"person", "people"}
        ]
        if not person_categories:
            raise ValueError("El JSON no contiene una categoria llamada person.")

        person_category = set(person_categories)
        annotations_by_image: Dict[int, List[dict]] = {}
        for annotation in coco.get("annotations", []):
            if annotation["category_id"] in person_category:
                annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

        self.images = [
            record
            for record in coco.get("images", [])
            if record["id"] in annotations_by_image
        ]
        self.annotations = annotations_by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        record = self.images[index]
        image_path = self.images_root / record["file_name"]
        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []
        for annotation in self.annotations[record["id"]]:
            x, y, width, height = annotation["bbox"]
            boxes.append([x, y, x + width, y + height])
            labels.append(1)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([record["id"]]),
        }
        if self.transforms:
            image = self.transforms(image)
        else:
            import torchvision.transforms as T
            image = T.PILToTensor()(image).float().div(255)
        return image, target

