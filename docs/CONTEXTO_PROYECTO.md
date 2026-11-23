# Contexto del proyecto: sistema de conteo y seguimiento de personas

## 1. Objetivo

Desarrollar un sistema de inteligencia artificial en Python que, a partir de la camara web o de un video, pueda:

- detectar personas mediante una red neuronal convolucional;
- seguir cada persona a lo largo de los fotogramas;
- contar entradas y salidas respecto de una linea virtual;
- calcular la ocupacion actual;
- consultar estadisticas de ocupacion mediante un asistente;
- mostrar visualmente las detecciones y el seguimiento durante una demostracion.

El sistema sera una herramienta de demostracion academica. No se utilizaran
reconocimiento facial, identificacion nominal ni almacenamiento de datos personales.

## 2. Alcance inicial

### Incluido

- Procesamiento de video MP4 o de la camara web.
- Deteccion de personas con una CNN.
- Asignacion de un identificador temporal a cada individuo detectado.
- Conteo cuando una persona cruza una linea configurada.
- Registro de entrada, salida y ocupacion historica.
- Interfaz grafica o visualizacion mediante OpenCV.
- Consigna de estadisticas por medio de comandos simples o una interfaz de terminal.
- Documentacion de instalacion, ejecucion y pruebas.

### No incluido inicialmente

- Reconocimiento de identidades.
- Analisis emocional o atributivo.
- Cobro, control de acceso o acciones legales.
- Altas cantidades de camaras distribuidas.
- Capacidad de reidentificar a una persona despues de una desconexion prolongada.

Este proyecto **DEBE** de utilizar un modelo **DESDE CERO**, es decir, que no haya sido preentrenado o descargado, la idea es realizar el proceso de crear nuestro propio modelo utilizando una CNN.
La idea es dividir el desarrollo entre el sistema que utiliza el modelo, y la creación y entrenamiento del modelo.
El dataset que utilizamos es probable que no tenga una carga por medio de codigo, es decir que dependemos de alimentar al modelo por medio de una carpeta llamada data o resources.

El modelo resultante para utilizar en este proyecto debe ser exportado y utilizado como tal.