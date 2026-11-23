import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class PascalVocPersonDataset(Dataset):
    """Dataset PASCAL VOC reducido a las cajas con nombre person."""

    def __init__(
        self,
        annotations_root: str,
        images_root: str,
        split_file: str | None = None,
        transforms=None,
    ):
        self.annotations_root = Path(annotations_root)
        self.images_root = Path(images_root)
        self.transforms = transforms

        if split_file:
            identifiers = [
                line.strip()
                for line in Path(split_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            identifiers = [
                path.stem
                for path in sorted(self.annotations_root.glob("*.xml"))
            ]

        self.records: List[dict] = []
        for identifier in identifiers:
            xml_path = self.annotations_root / f"{identifier}.xml"
            if not xml_path.exists():
                continue

            try:
                root = ET.parse(xml_path).getroot()
            except ET.ParseError as error:
                raise ValueError(f"XML invalido: {xml_path}") from error

            filename = root.findtext("filename")
            if not filename:
                filename = f"{identifier}.jpg"

            boxes = []
            for object_node in root.findall("object"):
                name = (object_node.findtext("name") or "").strip().lower()
                if name != "person":
                    continue

                box_node = object_node.find("bndbox")
                if box_node is None:
                    continue
                try:
                    xmin = float(box_node.findtext("xmin"))
                    ymin = float(box_node.findtext("ymin"))
                    xmax = float(box_node.findtext("xmax"))
                    ymax = float(box_node.findtext("ymax"))
                except (TypeError, ValueError):
                    continue

                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])

            if boxes:
                image_path = self.images_root / filename
                if image_path.exists():
                    self.records.append({"image_path": image_path, "boxes": boxes})

        if not self.records:
            raise ValueError(
                "No se encontraron imagenes con cajas de la clase person "
                "en las anotaciones XML indicadas."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("RGB")

        boxes = torch.as_tensor(record["boxes"], dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": torch.ones((boxes.shape[0],), dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }
        if self.transforms:
            image = self.transforms(image)
        else:
            import torchvision.transforms as T
            image = T.PILToTensor()(image).float().div(255)
        return image, target
