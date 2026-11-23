# Datos

Coloque los archivos del dataset en `data/raw/`. Los contenidos de `data/` no
se incluyen en el repositorio; solo se conserva este README.

El proyecto admite dos formatos de anotacion.

## PASCAL VOC XML

El dataset observado utiliza esta estructura:

```text
data/raw/Train/
  Annotations/
    2008_000008.xml
    img (100).xml
  JPEGImages/
    2008_000008.jpg
    img (100).jpg
  train.txt
```

Cada XML contiene cajas y nombres de clase. La clase `person` se usa como
objetivo; objetos como `nonhuman` se ignoran.

El entrenamiento puede ejecutarse directamente:

```powershell
python scripts/train.py
```

Tambien admite rutas explicitas:

```powershell
python scripts/train.py --annotations data/raw/Train/train.txt --images data/raw/Train/JPEGImages
python scripts/train.py --annotations data/raw/Train/Annotations --images data/raw/Train/JPEGImages
```

Cuando `train.txt` indica los identificadores, solo se cargan esas imagenes.
Las coordenadas XML se convierten automaticamente a cajas `[xmin, ymin, xmax, ymax]`.

## COCO JSON

Formato alternativo recomendado:

```text
data/
  raw/
    images/
    annotations/
      train.json
      val.json
```

Las categorias deben representar personas con un identificador de categoria
igual a `1`. Campos requeridos:

- `images`: `id`, `file_name`, `width`, `height`
- `annotations`: `image_id`, `category_id`, `bbox` (`[x, y, ancho, alto]`)
- `categories`: `id=1`, `name=person`

Convencion: los archivos JSON pueden llamarse `train.json` y `val.json`, pero
los argumentos de entrenamiento permiten indicar otros nombres.

