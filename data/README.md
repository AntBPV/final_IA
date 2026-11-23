# Datos

Coloque los archivos del dataset en `data/raw/`. Los contenidos de `data/` no
se incluyen en el repositorio; solo se conserva este README.

El proyecto admite dos formatos de anotacion.

## PASCAL VOC XML

El dataset observado utiliza esta estructura:

```text
data/raw/train/
  Annotations/
    2008_000008.xml
    img (100).xml
  JPEGImages/
    2008_000008.jpg
    img (100).jpg
  train.txt
data/raw/val/
  Annotations/
  JPEGImages/
  val.txt
data/raw/Test/
  Annotations/
  JPEGImages/
  test.txt
```

Cada XML contiene cajas y nombres de clase. La clase `person` se usa como
objetivo; objetos como `nonhuman` se ignoran.

El entrenamiento puede ejecutarse directamente:

```powershell
python scripts/train.py
```

Tambien admite rutas explicitas:

```powershell
python scripts/train.py --annotations data/raw/train/train.txt --images data/raw/train/JPEGImages
python scripts/train.py --annotations data/raw/train/Annotations --images data/raw/train/JPEGImages
```

Cuando `train.txt` indica los identificadores, solo se cargan esas imagenes.
Las coordenadas XML se convierten automaticamente a cajas `[xmin, ymin, xmax, ymax]`.
El conjunto `val` tiene el mismo formato y se usa para decidir el mejor
estado del entrenamiento. No debe usarse para ajustar manualmente cada cambio
durante varias iteraciones sin reservar otro conjunto.

## Interpretacion de train, val y test

- `train`: ajusta los pesos del modelo.
- `val`: selecciona epocas, umbrales y decisiones de entrenamiento.
- `test`: debe reservarse para la prueba final y no utilizarse para ajustar el
  modelo.

La carpeta `val` no significa validacion fisica ni prueba; es una particion
estandar usada durante el desarrollo. La carpeta `Test` esta disponible para
la evaluacion final; su nombre completo no debe confundirse con el conjunto
de entrenamiento.

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

