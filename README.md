# Sistema academico de conteo y seguimiento de personas

Proyecto en Python para detectar personas desde una webcam o un video, asignar
identificadores temporales, contar cruces de una linea virtual y consultar la
ocupacion historica. El modelo es una red neuronal creada desde cero; no usa
pesos preentrenados.

Este repositorio contiene la estructura y el esqueleto ejecutable del proyecto.
No incluye el dataset ni modifica el contexto academico en `docs/`.

La metodologia, arquitectura, estado tecnico y limitaciones del detector
actual se documentan en [docs/MODELO_TECNICO.md](docs/MODELO_TECNICO.md).

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

El comando anterior usa `data/raw/train/` como entrenamiento y
`data/raw/val/` como validacion. Conserva el mejor estado segun la perdida
de validacion.

Para usar COCO:

```powershell
python scripts/train.py --annotations data/raw/annotations/train.json --images data/raw/images
```

El script produce `models/person_detector.pth` y un exporte
`models/person_detector.torchscript`. El entrenamiento puede tardar
considerablemente sin GPU.

Evaluar el modelo exportado sobre la particion de validacion:

```powershell
python scripts/evaluate.py
```

Para evaluar una sola vez sobre la particion final reservada:

```powershell
python scripts/evaluate.py --partition test
```

No use `--partition test` para ajustar épocas, umbrales o arquitectura.
Primero ajuste el modelo con `train/` y controle el progreso con `val/`.

El informe incluye el numero de imagenes, cajas reales, detecciones y
`mAP@0.50`. No use `test` hasta tener una particion independiente reservada
exclusivamente para evaluacion final.

## Demostracion

Con una webcam:

```powershell
python scripts/run_demo.py --source 0 --model models/person_detector.torchscript
```

Con un archivo MP4:

```powershell
python scripts/run_demo.py --source video.mp4 --model models/person_detector.torchscript
```

La demostracion puede reducir la resolucion y ejecutar inferencia cada varios
fotogramas:

```powershell
python scripts/run_demo.py --source 0 --width 640 --infer-every 3 --threshold 0.35
```

Para diagnosticar aceleracion disponible:

```powershell
python scripts/check_devices.py
```

Dentro del entorno `.env-dml`, comprobar el modelo real sobre DirectML:

```powershell
python scripts/check_directml.py
```

Para crear un entorno separado de prueba con AMD DirectML y Python 3.12:

```powershell
py -3.12 -m venv .env-dml
.\.env-dml\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-amd.txt
python scripts/check_devices.py
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

