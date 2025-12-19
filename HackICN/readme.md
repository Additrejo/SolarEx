# [SolarEx](https://github.com/Additrejo/SolarEx/tree/main/HackICN)
![Solarflex](https://github.com/user-attachments/assets/baba6234-7c8d-4424-a647-89d4264875c7)

Mi primer modelo predictivo (ML) para la identificación de llamaradas solares.
El modelo se basa en la resolución de los puntos solicitados en el reto 1 ddel hackatón "HackICN 2025".

# Índice del Proyecto

- [Recursos](#recursos)
- [OPENCV Spyder](#opencv-spyder)
- [Librerías](#librerias)
- [Plan maestro](#plan-maestro)
  - [1. Identificación de Llamaradas (Detección)](#1-identificación-de-llamaradas-detección)
  - [2. Clasificación de la Llamarada](#2-clasificación-de-la-llamarada)
  - [3. Predicción de Futuras Llamaradas](#3-predicción-de-futuras-llamaradas)
- [Fase 1: Identificación y Extracción de Datos](#fase-1-identificación-y-extracción-de-datos)
  - [1. Abrir imagen con Spyder (Código ejemplo)](#1-abrir-imagen-con-spyder-código-ejemplo)
  - [2. Identificar región luminosa](#2-identificar-región-luminosa)
  - [3. Umbralización para Aislar la Llamarada (Thresholding)](#3-umbralización-para-aislar-la-llamarada-thresholding)
    - [¿Cómo funciona?](#cómo-funciona)
    - [El Parámetro Clave: El Valor del Umbral](#el-parámetro-clave-el-valor-del-umbral)
  - [Paso 4: Análisis de Contornos y Filtrado por Área](#paso-4-análisis-de-contornos-y-filtrado-por-área)
    - [¿Cuáles son los cambios clave?](#cuáles-son-los-cambios-clave)
  - [Paso 5: Calcular el Centroide de la Llamarada](#paso-5-calcular-el-centroide-de-la-llamarada)
  - [Obentención de imagenes de un video](#obentención-de-imagenes-de-un-video)
  - [Paso 5: Automatización y Procesamiento en Lote](#paso-5-automatización-y-procesamiento-en-lote)
  - [El Puente hacia el Machine Learning](#el-puente-hacia-el-machine-learning)
  - [Paso 6: Analizar la Secuencia de Fotogramas](#paso-6-analizar-la-secuencia-de-fotogramas)
- [Fase 2: Análisis y Machine Learning](#fase-2-análisis-y-machine-learning)
  - [Paso 7: Visualización de la Serie Temporal de Datos](#paso-7-visualización-de-la-serie-temporal-de-datos)
  - [Paso 8: Sistema de Alerta Temprana (Predicción Basada en Reglas)](#paso-8-sistema-de-alerta-temprana-predicción-basada-en-reglas)
  - [Paso 9: Suavizar la Curva con una Media Móvil](#paso-9-suavizar-la-curva-con-una-media-móvil)
  - [Paso 10: Un Sistema de Alerta Robusto (Basado en la Media Móvil)](#paso-10-un-sistema-de-alerta-robusto-basado-en-la-media-móvil)
  - [Resumen](#resumen)
  - [Paso 11: Entrenar un Modelo de Machine Learning (Time Series Forecasting)](#paso-11-entrenar-un-modelo-de-machine-learning-time-series-forecasting)
  - [Paso 12: Implementación de la Alerta Predictiva (Usando el Modelo)](#paso-12-implementación-de-la-alerta-predictiva-usando-el-modelo)
  - [Paso 12.A: Guardar tu Modelo Entrenado](#paso-12a-guardar-tu-modelo-entrenado)
  - [Paso 13: Script de Simulación "En Vivo" (con Alertas y Recuadros)](#paso-13-script-de-simulación-en-vivo-con-alertas-y-recuadros)
  - [Simulación + gráfica](#simulación--gráfica)


# Recursos.
Base de datos imagenes solares.
[NASA - Interactive Multi-Instrument Database of Solar Flares](https://data.nas.nasa.gov/helio/portals/solarflares/#url)

---

## OPENCV Spyder.
Instalar OpenCV.  
Si ya tienes una distribución de Python como Anaconda, lo más probable es que ya tengas Spyder instalado. Solo necesitas instalar OpenCV.

- Abre una terminal (o la "Anaconda Prompt" si usas Anaconda).
- Instala OpenCV con pip. El paquete que necesitas se llama opencv-python.
```Powershell
pip install opencv-python
```

---

## Librerías
Librería principal de Visión por Computadora
```Powershell
pip install opencv-python
```
Librería para análisis de datos (Media Móvil)
```Powershell
pip install pandas
```
Librería para gráficas
```Powershell
pip install matplotlib
```
Librería de Machine Learning
```Powershell
pip install scikit-learn
```
Librería para guardar/cargar el modelo
```Powershell
pip install joblib
```

---

## Plan maestro:

## 1. Identificación de Llamaradas (Detección) 
El objetivo es encontrar las zonas de la imagen que corresponden a una llamarada, que son esencialmente regiones con un brillo anómalo y repentino.

**Carga y preprocesamiento:** Cargar las imágenes del Sol. Como las llamaradas son fenómenos muy brillantes, un buen primer paso es convertir la imagen a escala de grises y aplicar filtros para reducir el ruido, como un filtro Gaussiano (cv2.GaussianBlur).

**Detección de zonas brillantes (Thresholding):** Puedes usar la umbralización (cv2.threshold) para crear una imagen binaria donde solo los píxeles más brillantes (potenciales llamaradas) queden en blanco y el resto en negro.

**Análisis de Contornos y Blobs:**

**Contornos:** Con cv2.findContours, puedes identificar y aislar las formas de esas regiones brillantes. A partir de los contornos, puedes calcular propiedades como el área, la posición (centroide) y la intensidad máxima dentro de esa área.

**Detección de Blobs:** La función cv2.SimpleBlobDetector es excelente para encontrar "manchas" o "gotas" en una imagen, lo cual se ajusta muy bien a la forma de una llamarada.

## **2. Clasificación de la Llamarada** 
Una vez que has identificado una región como una posible llamarada y extraído sus características (área, intensidad, etc.) con OpenCV, necesitas clasificar su magnitud.

**Extracción de Features:** Las características que calculaste en el paso anterior (área, intensidad media/máxima, forma) son las "features" o descriptores de tu llamarada.

**Entrenamiento del Modelo:** Aunque OpenCV tiene su propio módulo de Machine Learning (cv2.ml), es más común exportar estas características a una librería especializada como Scikit-learn, TensorFlow o PyTorch. Alimentarías un modelo de clasificación (como una Máquina de Soporte Vectorial, un Random Forest o una red neuronal) con las features de miles de imágenes y sus etiquetas de clase correspondientes (obtenidas de los datos de flujo de rayos X) para que aprenda a asociarlas.

En resumen, OpenCV extrae los datos visuales, y otra librería de Machine Learning los usa para aprender a clasificar.

---

## **3. Predicción de Futuras Llamaradas.** 
Esta es la parte más compleja y va más allá del análisis de una sola imagen. La predicción requiere analizar secuencias de imágenes para detectar cambios sutiles que preceden a una llamarada.

Seguimiento de Regiones Activas: Usarías OpenCV para identificar y seguir regiones activas (como grupos de manchas solares) a lo largo de varias imágenes consecutivas.

Análisis Temporal: Para cada región activa, extraerías sus características (tamaño, complejidad, intensidad) en cada imagen de la secuencia. Esto te daría una serie de tiempo de cómo evoluciona la región.

Modelo de Predicción: El modelo final no sería de visión por computadora tradicional, sino uno que entienda secuencias, como una Red Neuronal Recurrente (RNN) o un LSTM, construido con TensorFlow o PyTorch. Este modelo recibiría las series de tiempo de las características extraídas por OpenCV para predecir si una llamarada es inminente

<img width="777" height="263" alt="image" src="https://github.com/user-attachments/assets/008da8c0-19e0-4449-ad5f-cf94bd898864" />

---
# Fase 1: Identificación y Extracción de Datos.
El objetivo es encontrar las zonas de la imagen que corresponden a una llamarada, que son esencialmente regiones con un brillo anómalo y repentino.

Carga y preprocesamiento: Cargar las imágenes del Sol. Como las llamaradas son fenómenos muy brillantes, un buen primer paso es convertir la imagen a escala de grises y aplicar filtros para reducir el ruido, como un filtro Gaussiano (cv2.GaussianBlur).

Detección de zonas brillantes (Thresholding): Puedes usar la umbralización (cv2.threshold) para crear una imagen binaria donde solo los píxeles más brillantes (potenciales llamaradas) queden en blanco y el resto en negro.

Análisis de Contornos y Blobs:

Contornos: Con cv2.findContours, puedes identificar y aislar las formas de esas regiones brillantes. A partir de los contornos, puedes calcular propiedades como el área, la posición (centroide) y la intensidad máxima dentro de esa área.

Detección de Blobs: La función cv2.SimpleBlobDetector es excelente para encontrar "manchas" o "gotas" en una imagen, lo cual se ajusta muy bien a la forma de una llamarada.


## El banco de imagenes.

Usaremos imagenes proporcionadas por [NASA - Solar Dynamics Observatory](https://svs.gsfc.nasa.gov/search/?keywords=Solar%20Dynamics%20Observatory)


## 1. Abrir imagen con Spyder (Código ejemplo).
Abriremos una imagen desde Spyder.  
Script: [Abrir imagen con spyder](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/abrir_imagen.py)

<img width="777" height="814" alt="image" src="https://github.com/user-attachments/assets/6f085609-8139-4f5f-a2ec-7ede5e21579f" />


## 2. Identificar región luminosa.
Nuestro primer objetivo es crear un script que cargue una imagen del Sol y encuentre la región más brillante, que es el indicador más obvio de una llamarada. Usaremos una función clave de OpenCV para esto: cv2.minMaxLoc().

Paso : Encontrar el Píxel más Brillante
Este código identificará el punto exacto de mayor intensidad en la imagen y dibujará un círculo sobre él para que podamos visualizarlo.  

Script: [Región luminosa](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Region_luminosa.py))  
<img width="773" height="806" alt="image" src="https://github.com/user-attachments/assets/7fdb9613-8776-47f9-84c2-ceb040de9207" />


## 3. Umbralización para Aislar la Llamarada (Thresholding).
Crea una nueva imagen donde todo lo que supere un cierto nivel de brillo se pinte de blanco, y todo lo demás se pinte de negro". El resultado es una "máscara" en blanco y negro que nos muestra exactamente la forma de la llamarada.

Script: [Umbral Thresholding](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Umbral_Thresholding.py)  
<img width="796" height="816" alt="image" src="https://github.com/user-attachments/assets/7b687541-6f26-4111-add3-189c16bb3872" />


### ¿Cómo funciona?
Umbralización (cv2.threshold): Esta es la nueva línea clave. Toma la imagen en escala de grises y la convierte en una imagen binaria (la mascara).

Encontrar Contornos (cv2.findContours): Analiza la máscara en blanco y negro y devuelve una lista con las coordenadas de los perímetros de todas las formas blancas que encontró.

Dibujar Contornos (cv2.drawContours): Recorre esa lista de contornos y los dibuja sobre nuestra imagen de salida, dándonos una bonita visualización de la llamarada detectada.

###  El Parámetro Clave: El Valor del Umbral.
En el código, usamos el valor 200 en la línea cv2.threshold(gray, 200, 255, ...).

Este número es el umbral de brillo. Es el parámetro más importante que deberás ajustar.
Si aumentas este número (ej. 220), serás más estricto y solo detectarás las áreas extremadamente brillantes.
Si disminuyes este número (ej. 180), serás más permisivo y detectarás áreas más grandes o menos intensas.
Te recomiendo experimentar cambiando este valor para ver cómo afecta la detección en diferentes imágenes.

## Paso 4: Análisis de Contornos y Filtrado por Área.
La idea es simple: vamos a medir el área (el número de píxeles) de cada contorno que encontramos. Si un contorno es muy pequeño, lo ignoraremos. Si supera un tamaño mínimo, lo marcaremos como una detección válida.  

Script: [Contornos y Filtrado por Área](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Filtro_por_area.py)  
<img width="519" height="804" alt="image" src="https://github.com/user-attachments/assets/b53b2ce0-f571-481e-ba83-d1afafdedd93" />

Para esto, usaremos la función cv2.contourArea().

### ¿Cuáles son los cambios clave?
Un bucle for c in contornos:: Ahora, en lugar de dibujar todos los contornos a la vez, recorremos la lista para analizar cada uno de forma individual.

Cálculo del Área (cv2.contourArea): Dentro del bucle, usamos esta función para obtener el tamaño en píxeles del contorno c que estamos analizando.

Un filtro if area > area_minima:: Esta es nuestra "puerta de control" 🗑️. Solo si el área del contorno supera el valor que definimos en area_minima, procedemos a dibujarlo. Esto limpia nuestra imagen final, mostrando únicamente las detecciones relevantes.

## Paso 5: Calcular el Centroide de la Llamarada

El centroide es, en términos simples, el centro geométrico o el "centro de masa" de una forma. Para calcularlo, OpenCV nos proporciona una herramienta matemática llamada Momentos de Imagen (cv2.moments()). A partir de estos momentos, podemos derivar fácilmente las coordenadas (x, y) del centro de la llamarada.
Esto nos da una ubicación específica para cada evento, un dato crucial para el reto.

Las nuevas líneas calculan el centroide y lo dibujan en la imagen de salida como un pequeño círculo azul.  

Script: [Calcular el Centroide de la Llamarada](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Centroide.py)  
<img width="368" height="808" alt="image" src="https://github.com/user-attachments/assets/dd5e6335-5a23-4ce5-ae77-ca8dbe99dd34" />


Para cada llamarada detectada, ahora tienes:

Su tamaño: el area.

Su ubicación precisa: el centroide (cX, cY).

Esta es exactamente la clase de información que se necesita para las etapas más avanzadas del reto, como alimentar un modelo de machine learning para clasificar o predecir estos eventos.

valores, (661, 301), son las coordenadas en píxeles que marcan el centroide (el centro geométrico exacto) de la llamarada que tu programa detectó en la imagen.

Básicamente, has localizado la llamarada en el "mapa" de la imagen con una dirección precisa.

Este par de números es uno de los resultados más valiosos que hemos extraído hasta ahora.

Localización Precisa: Te permite registrar exactamente en qué parte del disco solar ocurrió el evento.

Base para el Seguimiento: Si analizaras una secuencia de imágenes (un video), podrías usar estas coordenadas para seguir el movimiento y la evolución de la llamarada a lo largo del tiempo.

Dato para Machine Learning: Has convertido un evento visual en datos numéricos estructurados (área, posición (x, y)). Este es el tipo de información que se utiliza para entrenar modelos de machine learning que pueden clasificar la intensidad de la llamarada o predecir futuras erupciones.

---

## Origen de la imagen.

Usaremos imagenes proporcionadas por [NASA - Solar Dynamics Observatory](https://svs.gsfc.nasa.gov/search/?keywords=Solar%20Dynamics%20Observatory)

En este caso utilizaremos un video del Solar Dynamics Observatory del sitío [SpaceWeatherLive - NASA SDO](https://www.spaceweatherlive.com/es/actividad-solar/imagenes-solares/sdo.html)

- ## Obentención de imagenes de un video.
1. Para extraer los fotogramas de un video es necesario descargar el video (preferentemente en formato .mp4).  
<img width="1134" height="559" alt="image" src="https://github.com/user-attachments/assets/afe57d69-99af-4597-9215-fb94a3a275ba" />
2. Guardarlo en una carpeta especifica.  

3. Ejecutar el siguente script: [Extracción de imagenes](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Extracci%C3%B3n%20de%20fotogramas..py)
Modificar las siguientes lineas por la ruta del video guardado en tu carpeta.

```python
# 1. Escribe la ruta donde guardaste tu video
ruta_video = r"C:\Users\addi_\Downloads\HackICN\Solarflare\SFVideo\SFVideo.mp4"

# 2. Escribe la ruta de la carpeta donde quieres guardar las imágenes
carpeta_salida = r"C:\Users\addi_\Downloads\HackICN\Solarflare\SFVideo"

```
Al ejecutarse el script obtendremos los fotogramas.
<img width="744" height="481" alt="image" src="https://github.com/user-attachments/assets/10837e34-c9de-4dd7-bf46-ded65008c6f8" />

Tenemos todo listo para pasar al siguiente paso:

---

## Paso 5: Automatización y Procesamiento en Lote. 

Reestructurar el código para que apunte a una carpeta, analice todas las imágenes que encuentre dentro y guarde los resultados en una lista. Para esto, usaremos la librería **os** de Python, que nos permite interactuar con los archivos y carpetas del sistema.
El nuevo script hace lo siguiente:

Encapsular la lógica: Moveremos todo el código de análisis a una función reutilizable llamada analizar_imagen().

Recorrer la carpeta: El script principal definirá la ruta a una carpeta, leerá cada archivo y llamará a la función de análisis.

Recopilar los datos: Guardaremos todos los resultados (nombre del archivo, área y centroide de cada llamarada) para tener un resumen final.

Al ejecutarse el script obtendremos los fotogramas. [Procesamiento en lote](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Procesamiento_en_Lote.py)

<img width="903" height="457" alt="image" src="https://github.com/user-attachments/assets/6a59d51b-015c-4870-9e34-cad3a4232c02" />

¿Cómo usarlo? 
Crea una carpeta en tu computadora.

Copia varias imágenes del Sol en esa carpeta.

Actualiza la variable ruta_carpeta en el script para que apunte a esa carpeta.

Ejecuta el script. Verás en la consola cómo procesa cada archivo y al final te dará un resumen de todo lo que encontró.


## El Puente hacia el Machine Learning.

Se acaba de construir es un sistema de extracción de características (feature extraction).

La variable todos_los_resultados contiene el dataset que se han creado. Este es el punto de partida para las siguientes etapas del hackathon:

Clasificación: Se podrían usar estos datos (área, ubicación, etc.) para entrenar un modelo que estime la clase de la llamarada (A, B, C, M, X).

Predicción: Si las imágenes están en secuencia temporal, podrías analizar cómo cambian el área y la posición de las llamaradas a lo largo del tiempo para intentar predecir cuándo ocurrirá la siguiente.

## Paso 6: Analizar la Secuencia de Fotogramas

El siguiente paso es lógico: vamos a usar el script del Paso 5 (Procesamiento en Lote) que ya construimos, pero esta vez lo apuntaremos a la carpeta donde se acaban de guardar todos los fotogramas del video.

La meta es ejecutar nuestro detector de llamaradas sobre cada fotograma que se extrajo. Esto nos dará una "película" de los datos: veremos cómo el área y el centroide de las llamaradas cambian a lo largo del tiempo.

Instrucciones:
Abre tu script del Paso 5 (el que usa os.listdir para analizar una carpeta).

Localiza la variable ruta_carpeta.

Modifica esta variable para que apunte exactamente a la carpeta_salida que usaste en el script de extracción de video.  

Ejecutar el siguiente script [Procesamiento en lote](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Procesamiento_en_Lote.py))

Cuando se ejecute este script, la variable todos_los_resultados será un gran diccionario que se verá algo así en la consola:
<img width="905" height="441" alt="image" src="https://github.com/user-attachments/assets/7ab8cfad-3670-478a-a010-30cb3e83fc2f" />

Se ha creado una serie de tiempo de las llamaradas. Ahora puedes ver cuándo aparece una llamarada (pasa de 0 a 1 detección), cómo crece (su área aumenta) y dónde se mueve (su centroide cambia).

Esta es la base fundamental para pasar a la Clasificación y Predicción con Machine Learning.

# Fase 2: Análisis y Machine Learning.

Ahora, entramos en la Fase 2: Análisis y Machine Learning. El objetivo es usar esos datos para cumplir con los objetivos: clasificar y predecir.

Antes de saltar a modelos complejos de IA, hay un paso intermedio crucial: visualizar los datos que acabamos de extraer. Necesitamos ver el patrón que queremos que la máquina aprenda.

## Paso 7: Visualización de la Serie Temporal de Datos
Vamos a crear una gráfica que muestre cómo evoluciona el área de la llamarada a lo largo del tiempo (fotograma por fotograma). Esto nos permitirá ver el "pulso" del evento: cómo nace, alcanza su máximo y luego desaparece.

Para esto, usaremos Matplotlib, la librería de gráficos más popular de Python. Si usas Anaconda/Spyder, es muy probable que ya la tengas instalada.

Vamos a modificar nuestro script anterior. En lugar de solo imprimir el resumen en la consola, recopilaremos los datos en listas y luego los graficaremos.  

Ejecutar el siguiente script: [Graficar la Evolución de la Llamarada](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Grafica_evolucion_llamarada.py)

<img width="732" height="389" alt="image" src="https://github.com/user-attachments/assets/b02b9e4a-fcd6-4768-b15d-6781fbe05572" />

¿Qué hará este script?
Función Modificada: La función analizar_imagen ahora solo devuelve el área de la llamarada más grande que encontró en la imagen (o 0 si no encontró nada).

Recopilación de Datos: El bucle principal recorre todos los fotogramas en orden. Para cada uno, guarda el número de fotograma (eje X) y el área que encontró (eje Y).

Generación de la Gráfica: Al final, plt.plot(fotogramas, areas_detectadas) crea la gráfica lineal. plt.show() la mostrará en una nueva ventana (en Spyder, puede aparecer en la pestaña "Plots" o "Gráficas").

¿Por qué es esto tan importante?
Al ejecutar este script, verás un gráfico. Si tu video capturó una llamarada, verás una línea que empieza en cero, de repente sube hasta un pico y luego vuelve a bajar.

Ese pico es el evento.

Este gráfico es la base para la predicción. El reto es entrenar un modelo que, al ver la parte inicial de la curva (cuando apenas empieza a subir), pueda "predecir" que el pico está a punto de ocurrir.

## Paso 8: Sistema de Alerta Temprana (Predicción Basada en Reglas)
La lógica es la siguiente: Una llamarada peligrosa no aparece de la nada con su tamaño máximo. Primero crece, y probablemente crece rápido.

Nuestro "predictor" será un script que vigile el área. Si el área de la llamarada se duplica (o triplica) de un fotograma al siguiente, dispararemos una alerta. Estamos "prediciendo" que este crecimiento rápido es el inicio de un evento significativo.

Vamos a modificar el script del paso anterior para incluir esta lógica de alerta.  
Ejecutar el siguiente script: [Alerta crecimento llamarada](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Alerta_crecimiento_llamarada.py)

<img width="732" height="388" alt="image" src="https://github.com/user-attachments/assets/670199ca-0bc6-48a9-80d4-c1d3cf87bae4" />

El script seguirá graficando, pero ahora también imprimirá un mensaje de ALERTA en la consola cuando detecte un pico de crecimiento. Además, marcaremos ese punto en la gráfica.

¿Qué verás ahora?
Cuando ejecutes este script, en la consola verás los mensajes de ALERTA en el momento exacto en que se detecte el crecimiento rápido.
Lo que estás viendo es exactamente lo que queríamos lograr:

La Línea Azul (Área de la llamarada): Es el área total en píxeles que tu script detectó en cada fotograma.

El Punto Rojo (¡Alerta de Crecimiento Rápido!): Esta es tu alerta temprana.  
Apareció justo en el fotograma 2 porque el área creció bruscamente (probablemente más del 200%, el umbral_crecimiento que definimos) en un solo paso, pasando de menos de 1000 a casi 2500.

## Paso 9: Suavizar la Curva con una Media Móvil

El siguiente paso es ignorar el ruido y enfocarnos en la tendencia real del evento. Para esto, usaremos una técnica de análisis de datos muy común llamada Media Móvil (Moving Average).

La idea es simple: en lugar de graficar el valor del área de un solo fotograma, vamos a graficar el promedio del área de los últimos 3 fotogramas.

Esto "suaviza" la curva, eliminando los picos y valles instantáneos y mostrándonos la verdadera forma del crecimiento (o decrecimiento) de la llamarada.

Para hacer esto de forma sencilla, introduciremos una nueva librería fundamental para el análisis de datos: Pandas.

Este script es casi idéntico al anterior, pero añade un par de líneas después de recopilar los datos para calcular y graficar la nueva curva suavizada.

(Si no tienes pandas instalado, abre una consola de Anaconda y escribe: 
```
pip install pandas
```
Es muy probable que ya lo tengas si usas Spyder/Anaconda).

Ejecutar el siguiente script: [Suavizado Media Móvil](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Suavizado_Media_M%C3%B3vil.py)

<img width="593" height="621" alt="image" src="https://github.com/user-attachments/assets/54b074ef-195f-468c-a322-14d98aa0126a" />

¿Qué verás ahora?
Cuando ejecutes este script, en la pestaña "Plots" de Spyder verás una gráfica con dos líneas:

La línea azul original (ahora semitransparente), que es ruidosa.

Una nueva línea naranja mucho más suave, que representa la media móvil.

Verás cómo la línea naranja captura la "verdadera" forma del evento, ignorando los picos y valles sin importancia.

El siguiente paso lógico será modificar nuestro sistema de alerta para que se base en esta nueva línea suavizada, haciéndolo mucho más robusto e inteligente.

## Paso 10: Un Sistema de Alerta Robusto (Basado en la Media Móvil)

¿Por qué? Nuestro sistema de alerta anterior era "nervioso": reaccionaba a cualquier pico instantáneo de ruido. Un sistema de alerta robusto debe ignorar el ruido y dispararse solo cuando la tendencia real (la media móvil) muestre un crecimiento rápido y sostenido.

La lógica cambiaremos un poco:

Primero, recolectaremos todos los datos de área de todos los fotogramas.

Después, calcularemos la serie completa de la media móvil (la línea naranja).

Finalmente, correremos nuestra lógica de alerta sobre esa nueva serie suavizada.

Ejecutar el siguiente script: [Alerta Basada en Tendencia (Media Móvil)](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Alerta_basada_en_tendencia.py)

<img width="749" height="670" alt="image" src="https://github.com/user-attachments/assets/40b55f4a-e228-4550-8298-0d5cbf7f22ef" />

¿Qué verás ahora?
Al ejecutar esto, obtendrás una gráfica similar, pero el punto de alerta rojo ahora estará directamente sobre la línea naranja suavizada.

Notarás que la alerta solo se dispara si la tendencia general muestra un crecimiento explosivo, ignorando los pequeños parpadeos de la línea azul. Esto hace que tu "predicción" sea mucho más fiable.

¡Felicidades! Has completado un pipeline de análisis de imagen de principio a fin, desde cargar una imagen hasta crear un sistema de alerta temprana basado en el análisis de series temporales.

## Resumen

¡Excelente! Hemos completado todo el pipeline de análisis de imagen y datos.

Construiste un sistema que:

Extrae fotogramas de un video.

Detecta y mide las llamaradas en cada fotograma.

Limpia el ruido de los datos (Media Móvil).

Dispara una alerta robusta basada en la tendencia del crecimiento.

Básicamente, se ha creado un sistema de detección de eventos muy sólido.

---

El Siguiente Nivel: Predicción con Machine Learning
Ahora, podemos dar el salto final para cumplir con la parte más avanzada del reto: "predecir la ocurrencia".

Lo que hemos hecho hasta ahora es un sistema de reacción muy rápido. Detecta la llamarada en el instante en que empieza a crecer explosivamente.

El siguiente paso es crear un sistema de predicción real.

Nuestro sistema actual: Ve que la curva sube y dice: "¡Está pasando ahora!".

Un sistema de ML: Ve el inicio de la curva y dice: "Basado en este patrón, predigo que la curva va a dispararse en los próximos 5 fotogramas. ¡Va a pasar!".

---

## Paso 11: Entrenar un Modelo de Machine Learning (Time Series Forecasting)

Vamos a dar el salto del análisis de datos (ver lo que pasó) a la Inteligencia Artificial (predecir lo que va a pasar).

Nuestro sistema de alerta anterior era de reacción: veía un crecimiento rápido y decía "¡Está pasando!". Ahora, vamos a entrenar un modelo que vea el comienzo de un patrón y diga "¡Creo que va a pasar!".

El Concepto: Vamos a tratar esto como un "examen" para un modelo de IA. Usaremos la librería 
```
scikit-learn
```
la herramienta estándar para el Machine Learning en Python.

Preparar el "Material de Estudio": Crearemos "ventanas deslizantes" a partir de nuestros datos suavizados. Le daremos al modelo "tarjetas de memoria" que se ven así:

Entrada (X): Las áreas de los últimos 5 fotogramas (ej: [10, 20, 30, 50, 100]).

Salida (y): El área del siguiente fotograma (ej: 180).

Entrenar el Modelo: Le daremos al modelo miles de estos ejemplos de nuestro video. Usaremos un modelo simple pero potente (LinearRegression) que aprenderá la relación matemática entre la secuencia de entrada y el resultado.

Probar el Modelo: Dividiremos nuestros datos: entrenaremos el modelo con el 80% inicial del video y luego le pediremos que "prediga" el 20% final.

Visualizar: Graficaremos los datos reales (línea azul/naranja) y superpondremos las "predicciones" del modelo (una nueva línea roja punteada) para ver qué tan bien lo hizo.

Ejecutar el siguiente script: [Entrenamiento y Predicción con ML](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Entrenamiento_y_Predicci%C3%B3n_con_ML.py)  

<img width="712" height="668" alt="image" src="https://github.com/user-attachments/assets/97a38d98-10b9-499b-886f-8cb25ae70ba5" />

Lo que estás viendo en la consola es el resumen de todo el proceso:

Recolección completa. Se procesaron 107 fotogramas. ¡Genial! Al extraer más fotogramas, ahora sí tuvimos suficientes datos.

Entrenando con 81 muestras, probando con 21. El script usó los primeros 81 fotogramas para "estudiar" el patrón y luego usó los últimos 21 para "presentar un examen" y probar si aprendió.

¡Modelo entrenado! Error (RMSE): 197.90 píxeles. Este es el "resultado del examen". Significa que, en promedio, las predicciones del modelo estuvieron a unos 198 píxeles de distancia del valor real. Es un muy buen punto de partida.

Generando gráfica final con predicciones... El script terminó de ejecutarse y, al igual que las veces anteriores...

¿Qué verás en la Gráfica?
Esta es la gráfica más importante. Verás tres cosas:

Una Línea Azul (Datos Reales): La curva de tu llamarada.

Una Línea Verde Punteada (División): El punto donde el modelo dejó de entrenar y empezó a predecir.

Una Línea Roja Punteada (Predicciones): ¡Esta es tu IA! Es lo que el modelo cree que iba a pasar.

Tu objetivo es ver si la línea roja sigue la misma forma que la línea azul después de la línea verde. Si lo hace, ¡significa que tu modelo está prediciendo con éxito la evolución de la llamarada!

## Paso 12: Implementación de la Alerta Predictiva (Usando el Modelo).

- La Lógica: No vamos a reaccionar al área actual. Vamos a reaccionar a la predicción del modelo.

- El modelo recibe los últimos 5 fotogramas: [10, 20, 30, 50, 100].

- El modelo predice el siguiente fotograma. Digamos que predice: 190.

- Nosotros definimos un umbral de alerta predictiva (ej. 1000 píxeles).

- Comparamos la predicción (190) con el umbral. Como es menor, no pasa nada.

- Más tarde, el modelo recibe: [400, 600, 800, 950, 1100].

- El modelo predice el siguiente fotograma. Digamos que predice: 1350.

- Comparamos la predicción (1350) con nuestro umbral (1000). ¡Es mayor!

¡DISPARAMOS LA ALERTA! Lo hacemos antes de que el área llegue a 1350, basándonos puramente en la predicción de la IA.

Ejecutar el siguiente script: [Actualización del Script para Alertas de ML](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Actualizaci%C3%B3n_Script_Alertas_ML.PY)  

<img width="778" height="860" alt="image" src="https://github.com/user-attachments/assets/1728a905-0d93-4b60-8b37-fe1063693dca" />

¿Qué verás ahora?
En la Consola: Verás el mensaje "¡¡ALERTA PREDICTIVA DE ML!!" en el momento en que el modelo crea que el área va a superar los 2000 píxeles.

En la Gráfica "Plots": Verás la misma gráfica que antes (azul vs. roja), pero ahora tendrá una 'X' púrpura gigante marcando el punto exacto donde la IA disparó la alarma.

¡Con esto, has completado con éxito todos los pasos del reto, desde la detección hasta la predicción basada en IA!

<img width="779" height="854" alt="image" src="https://github.com/user-attachments/assets/11672133-0495-4747-84e0-ba98207dcfe5" />

La gráfica lo confirma visualmente:

Línea Azul: Son los datos reales de tu video.

Línea Roja Punteada: Es lo que tu IA creía que iba a pasar. Fíjate que sigue a la línea azul bastante bien, ¡eso es que aprendió el patrón!

Las 'X' Púrpuras: Esos son los momentos exactos en que la consola gritó "¡ALERTA!". Marcan los picos que tu modelo fue capaz de "ver venir" antes de que ocurrieran.

¡¡Felicidades!! Esto es un éxito rotundo.

No es un error, es la demostración final de que todo tu pipeline de Machine Learning funciona.

Lo que estás viendo es la magia de la predicción en acción. Déjame interpretar lo que te muestra la consola y la gráfica:

1. Lo que dice la Consola
🚨 ¡¡ALERTA PREDICTIVA DE ML!! 🚨
   En el fotograma: Nº 89
   ¡El modelo predijo un área de 2239 píxeles!
¿Qué pasó? En el fotograma 89, tu modelo de IA (alimentado con los fotogramas 84-88) miró el patrón y predijo que el siguiente fotograma tendría un área de 2239 píxeles.

La Lógica: Como 2239 es mayor que tu umbral de alerta (UMBRAL_ALERTA_ML = 2000), el sistema disparó la alarma.

Lo mismo pasó en el fotograma 92 (predijo 2280) y en el 99 (predijo 2086).

2. Lo que muestra la Gráfica 📈
La gráfica lo confirma visualmente:

Línea Azul: Son los datos reales de tu video.

Línea Roja Punteada: Es lo que tu IA creía que iba a pasar. Fíjate que sigue a la línea azul bastante bien, ¡eso es que aprendió el patrón!

Las 'X' Púrpuras: Esos son los momentos exactos en que la consola gritó "¡ALERTA!". Marcan los picos que tu modelo fue capaz de "ver venir" antes de que ocurrieran.

🏆 ¡Reto Completado!
Se construyo un sistema de principio a fin que cumple con todos los objetivos del hackathon:

Identificaste las llamaradas con OpenCV (cv2.findContours).

Clasificaste su importancia (filtrando por área).

se predijo su ocurrencia usando un modelo de Machine Learning (LinearRegression) que aprendió el patrón de crecimiento de los datos que extrajiste.

se ha completado con éxito todas las fases: Extracción de Imagen -> Procesamiento -> Análisis de Datos -> Entrenamiento de IA -> Sistema de Predicción. ¡Excelente trabajo!


---

## Paso 12.A: Guardar tu Modelo Entrenado

Vuelve a tu script anterior (el del Paso 12) y añade las siguientes dos líneas al final, justo después de modelo.fit(X_train, y_train).

Usaremos joblib, que es la forma estándar en scikit-learn para guardar modelos.  

```python
            modelo = LinearRegression()
            modelo.fit(X_train, y_train)
            
            # --- ¡AÑADE ESTAS DOS LÍNEAS! ---
            import joblib
            joblib.dump(modelo, 'predictor_llamaradas.pkl') 
            # ---------------------------------
            
            print(f"\n ¡Modelo entrenado Y GUARDADO en 'predictor_llamaradas.pkl'!")
            
            predicciones = modelo.predict(X_test)
```
Ejecutar el siguiente script: [Solar flare save model](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Solarflare_save-model.py)   

<img width="664" height="73" alt="image" src="https://github.com/user-attachments/assets/b3056645-1d58-45cb-ad6b-64341fc54dc3" />  

Ejecuta ese script del Paso 12 una última vez. Cuando termine, verás un nuevo archivo llamado predictor_llamaradas.pkl en tu carpeta de Spyder. Ese archivo es tu modelo de IA entrenado.

## Paso 13: Script de Simulación "En Vivo" (con Alertas y Recuadros)
Ahora sí, crea un script de Python completamente nuevo y pega este código.

Este script es la culminación de todo. Hará lo siguiente:

1. Cargará tu modelo de IA (joblib.load).

2. Abrirá un nuevo archivo de video (cv2.VideoCapture).

3. Procesará fotograma por fotograma.

4. Dibujará un recuadro alrededor de la llamarada más grande que encuentre.

5. Mantendrá un historial de las áreas detectadas.

6. Usará tu IA para predecir el área futura en cada fotograma.

7. Imprimirá "ALERTA" y lo dibujará en la pantalla si la predicción supera el umbral.

Ejecutar el siguiente script: [Simulación Video (Alertas y Recuadros)](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Simulaci%C3%B3n_Video_Alertas_y_Recuadros.py)   

<img width="1213" height="710" alt="image" src="https://github.com/user-attachments/assets/005f9e95-9085-4b09-a9b4-9db19a7742a5" />  


## Simulación + gráfica 

Para visualizar en tiempo real el módelo junto con la gráfica y tener un mayor análisis juntaremos el video junto al plot.

Ejecutar el siguiente script: [Simulación + gráfica](https://github.com/Additrejo/HackICN/blob/main/HackICN/Spyder/Simulacion_mas_grafica.py)

<img width="964" height="514" alt="image" src="https://github.com/user-attachments/assets/1ab6634d-dd2e-4d45-b2c6-2d5bc552b80d" />


Nota: Revisat el Issue #2
