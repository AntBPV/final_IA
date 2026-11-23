from PIL import Image

from src.modeling.voc_dataset import PascalVocPersonDataset


def test_voc_dataset_filters_nonhuman_and_converts_boxes(tmp_path):
    images = tmp_path / "JPEGImages"
    annotations = tmp_path / "Annotations"
    images.mkdir()
    annotations.mkdir()

    Image.new("RGB", (320, 240)).save(images / "sample.jpg")
    (annotations / "sample.xml").write_text(
        """<annotation>
        <filename>sample.jpg</filename>
        <object><name>person</name><bndbox>
        <xmin>10</xmin><ymin>20</ymin><xmax>100</xmax><ymax>200</ymax>
        </bndbox></object>
        <object><name>nonhuman</name><bndbox>
        <xmin>120</xmin><ymin>30</ymin><xmax>220</xmax><ymax>210</ymax>
        </bndbox></object>
        </annotation>""",
        encoding="utf-8",
    )

    dataset = PascalVocPersonDataset(str(annotations), str(images))
    image, target = dataset[0]

    assert image.shape[0] == 3
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [1]


def test_voc_dataset_keeps_negative_image_with_empty_boxes(tmp_path):
    images = tmp_path / "JPEGImages"
    annotations = tmp_path / "Annotations"
    images.mkdir()
    annotations.mkdir()

    Image.new("RGB", (320, 240)).save(images / "negative.jpg")
    (annotations / "negative.xml").write_text(
        """<annotation>
        <filename>negative.jpg</filename>
        <object><name>nonhuman</name><bndbox>
        <xmin>10</xmin><ymin>20</ymin><xmax>100</xmax><ymax>200</ymax>
        </bndbox></object>
        </annotation>""",
        encoding="utf-8",
    )

    dataset = PascalVocPersonDataset(str(annotations), str(images))
    _image, target = dataset[0]

    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
