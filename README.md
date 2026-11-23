# Sistema academico de conteo y seguimiento de personas

Proyecto en Python para detectar personas desde una webcam o un video, asignar
identificadores temporales, contar cruces de una linea virtual y consultar la
ocupacion historica. El modelo es una red neuronal creada desde cero; no usa
pesos preentrenados.

Este repositorio contiene la estructura y el esqueleto ejecutable del proyecto.
No incluye el dataset ni modifica el contexto academico en `docs/`.

## Estructura

```text
data/
  raw/
  README.md
docs/
models/
src/
  app/
  counting/
  modeling/
  statistics/
  tracking/
scripts/
tests/
```

## Instalacion

Desde Windows PowerShell:

```powershell
python -m venv .env
.\.env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Preparacion del dataset

Coloque las imagenes y anotaciones dentro de `data/raw/`. El proyecto admite:

- PASCAL VOC XML, como `data/raw/Train/Annotations/` y
  `data/raw/Train/JPEGImages/`, con `train.txt`.
- COCO JSON, mediante `--annotations` y `--images`.

Los archivos dentro de `data/` estan excluidos de Git. Consulte
`data/README.md` para el formato requerido.

## Entrenamiento

```powershell
python scripts/train.py
```

El comando anterior usa el dataset VOC existente en `data/raw/Train`. Para usar
COCO:

```powershell
python scripts/train.py --annotations data/raw/annotations/train.json --images data/raw/images
```

El script produce `models/person_detector.pth` y un exporte
`models/person_detector.torchscript`. El entrenamiento puede tardar
considerablemente sin GPU.

## Demostracion

Con una webcam:

```powershell
python scripts/run_demo.py --source 0 --model models/person_detector.torchscript
```

Con un archivo MP4:

```powershell
python scripts/run_demo.py --source video.mp4 --model models/person_detector.torchscript
```

Controles:

- `q`: salir
- `l`: colocar o reiniciar la linea con dos clics de raton
- `r`: resetear conteos

Consultar estadisticas:

```powershell
python scripts/assistant.py ocupacion
python scripts/assistant.py entradas
python scripts/assistant.py salidas
```

## Pruebas

```powershell
python -m pytest tests
```

