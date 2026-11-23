# Documentacion tecnica del modelo detector de personas

## Estado actual

Se dispone de dos artefactos:

- `models/person_detector.pth`: pesos y estado del modelo PyTorch.
- `models/person_detector.torchscript`: modelo exportado para inferencia.

El modelo puede cargarse y ejecutarse, por lo que constituye una primera
linea base entrenada y exportada. No debe considerarse validado cientificamente
hasta disponer de metricas sobre un conjunto de validacion independiente.

Una prueba de humo del artefacto exportado pudo cargarlo y ejecutar inferencia
sobre `2008_000008.jpg`, devolviendo las claves `boxes`, `labels` y `scores`.
En esa imagen no produjo detecciones por encima del umbral interno de Faster
R-CNN. Este resultado aislado no demuestra que el modelo haya fallado o tenga
una precision determinada; demuestra la necesidad de una evaluacion sistematica.

## Metodo de aprendizaje

La solucion utiliza aprendizaje supervisado:

- Se presentan imagenes y cajas delimitadoras reales.
- La red aprende a distinguir la clase `person` de un fondo o de una clase
  negativa.
- El optimizador ajusta los pesos para reducir las perdidas de clasificacion,
  regresion de cajas y propuestas de region.
- El proceso no utiliza pesos descargados ni inicializaciones preentrenadas.

El metodo es desde cero en el sentido requerido: `weights=None` para el
backbone. Sin embargo, el proceso utiliza arquitectura y optimizadores
construidos por PyTorch. Esto no implica aprender una arquitectura nueva desde
cero; significa que no se importan pesos de una red entrenada externamente.

## Arquitectura

La arquitectura es Faster R-CNN:

- Backbone: ResNet-50 con Feature Pyramid Network.
- Cabeza regional: Faster R-CNN con Region Proposal Network.
- RPN: genera regiones candidatas en distintas escalas.
- ROI/Head: clasifica candidatos y ajusta las cajas.
- Clases internas: 2, correspondientes a fondo y `person`.
- Umbral predeterminado de inferencia: 0.5.

Faster R-CNN se selecciono porque realiza deteccion de objetos en una sola
imagen mediante regiones candidatas y devuelve cajas delimitadoras, a diferencia
de una CNN clasica que solo produciria una etiqueta por imagen completa.

## Dataset observado

La estructura utilizada es PASCAL VOC:

```text
data/raw/train/
  Annotations/
  JPEGImages/
  train.txt
data/raw/val/
  Annotations/
  JPEGImages/
  val.txt
```

Se encontraron:

- 944 identificadores unicos en `train.txt`.
- 944 archivos XML validos analizados.
- 1,106 objetos etiquetados como `person`.
- 960 objetos etiquetados como `nonhuman`.

El cargador conserva las cajas cuyo nombre sea exactamente `person`, convierte
las coordenadas VOC de `xmin/ymin/xmax/ymax` y descarta la clase `nonhuman`.
El archivo de particion contiene 944 identificadores. El cargador conserva
ahora tambien las imagenes negativas, pero debe existir su XML e imagen.
En el dataset actual se cargaron 933 imagenes de entrenamiento y 160 de
validacion. De las 160 de validacion, 67 no contienen cajas de persona.

Las imagenes negativas se cargan con cajas vacias y etiquetas vacias. Esto
permite entrenar y evaluar falsos positivos sin mezclar las clases adicionales
como `nonhuman`.

## Particion y preprocesamiento

Actualmente el entrenamiento utiliza el conjunto indicado por `train.txt`
como conjunto completo de entrenamiento. No existe una particion fija de
validacion y no se calcula precision, recall, AP o IoU durante el entrenamiento.

Las transformaciones vigentes son:

- conversion de PIL a tensor;
- conversion a valores en punto flotante;
- normalizacion por division entre 255.

No se aplican aumentos de datos, recorte aleatorio, rotacion, cambio de color
ni escalado manual en el script actual. Faster R-CNN aplica internamente su
transformacion de escala de entrada.

## Configuracion de entrenamiento

Los valores configurados son:

- epocas predeterminadas: 50;
- tamano de lote: 2;
- tasa de aprendizaje: 0.005;
- momentum: 0.9;
- weight decay: 0.0005;
- optimizador: SGD;
- dispositivo: GPU CUDA si esta disponible; en caso contrario CPU;
- workers de datos: 0;
- clases internas: fondo y persona.

No se utiliza:

- planificador de tasa de aprendizaje;
- early stopping;
- validacion por epoca;
- mezcla de datos;
- aumentos de imagen;
- pesos preentrenados.

El `data loader` mezcla aleatoriamente las muestras con `shuffle=True`.

## Perdida

El entrenamiento suma las perdidas que devuelve Faster R-CNN, incluyendo:

- clasificacion de objetos de RPN;
- ajuste de cajas de RPN;
- clasificacion regional;
- regresion de cajas regional.

Se imprime la suma promedio de estas perdidas por epoca, pero no debe
interpretarse como precision ni accuracy. Es una medida de optimizacion
utilizada para observar si el proceso converge.

## Exportacion

El checkpoint se guarda mediante `torch.save(model.state_dict())`.
Para la inferencia se carga el modelo desde cero, se recupera el checkpoint,
se coloca en modo evaluacion y se exporta con `torch.jit.trace`.

El wrapper de exportacion devuelve directamente un diccionario con:

- `boxes`;
- `labels`;
- `scores`.

El exporte permitio ejecutar una prueba de carga e inferencia, pero el trazado
puede ser menos flexible ante cambios dinamicos de forma que una red
exportada mediante metodos basados en grafos completos.

## Limitaciones conocidas

- No existe conjunto de validacion separado.
- No se reportan metricas objetivas de deteccion.
- No hay curvas de perdida persistidas ni historial por epoca.
- No hay division reproducible train/validation/test configurada.
- No se evaluaron clases negativas, oclusion ni condiciones adversas.
- No existe evaluacion de precision temporal ni seguimiento.
- El rastreo y conteo descritos por el proyecto aun no forman parte de una
  integracion validada con este detector.
- Una prueba aislada no representa el rendimiento general.

## Criterios recomendados antes de integrarlo

Antes de desarrollar el sistema final alrededor del modelo se recomienda:

1. Crear particiones reproducibles train, validation y test.
2. Mantener imagenes negativas y casos dificiles en las particiones apropiadas.
3. Evaluar sobre test con precision, recall, mAP@0.5 y mAP@[0.50:0.95].
4. Revisar precision del seguimiento mediante pruebas de video.
5. Medir fps e uso de CPU/GPU con entradas representativas.
6. Elegir el umbral de confianza segun precision y recall requeridos.
7. Guardar cada ejecucion con epoca, perdida, particion, semilla y version
   del modelo.

## Conclusion de preparacion

El modelo esta listo como **baseline experimental**:

- el entrenamiento puede reproducirse;
- el checkpoint existe;
- el artefacto TorchScript puede cargarse;
- el cargador puede inferir;
- no hay evidencia todavia de precision valida fuera de la muestra de
  entrenamiento.

Por tanto, puede comenzar el desarrollo del sistema alrededor del modelo,
pero el detector no debe declararse concluido ni medir desempeno academico
hasta completar la evaluacion fuera de la particion de entrenamiento.

## Aceleracion AMD en Windows

El equipo utiliza una AMD Radeon RX 9060 XT, pero la comprobacion actual
confirmo que este entorno contiene una build CPU de PyTorch:

- `torch.cuda.is_available()` devuelve `False`;
- `torch.version.cuda` es `None`;
- no significa que la Radeon este defectuosa;
- significa que PyTorch no fue compilado para este dispositivo.

PyTorch oficial para Windows no utiliza ROCm de forma nativa. NVIDIA CUDA no
es compatible con una Radeon. Para ejecutar este proyecto con aceleracion AMD
en Windows existen caminos limitados:

- DirectML mediante `torch-directml`;
- una compilacion alternativa o otro runtime compatible;
- Windows ML/ONNX Runtime DirectML para una red exportada;
- WSL/LINUX con ROCm, sujeto a compatibilidad real de la RX 9060 XT.

El soporte DirectML de PyTorch esta disponible como preview y depende de Python,
PyTorch, controlador y operadores soportados. Faster R-CNN usa operaciones de
region proposal y ROI que no necesariamente estan implementadas igual en
DirectML. Para este proyecto se agrego seleccion opcional `--device dml`, pero
se debe considerar experimental hasta ejecutar una prueba real.

El entorno actual usa Python 3.12.0, mientras que `torch-directml` no ofrece
una instalacion compatible descubierta para esta version. Se recomienda crear
otro entorno Python 3.10 para probarlo:

```powershell
py -3.10 -m venv .env-dml
.\.env-dml\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements-amd.txt
python scripts/check_devices.py
```

Las versiones exactas de PyTorch y torchvision deben coincidir con las
dependencias aceptadas por la build de `torch-directml`. No se deben mezclar
el entorno CPU actual y el entorno DirectML.

Cuando CUDA o DirectML no este disponible, el sistema sera explicitamente
CPU. La demo usa una resolucion configurable y inferencia espaciada para
mantener una frecuencia razonable mientras no haya aceleracion compatible.
