# SolarEx - Análisis de imágenes del Sol para identificar y predecir llamaradas solares.

<!--------------------- PROYECTO -------------------------->

![Proyecto](https://img.shields.io/badge/Proyecto-SolarEx%20-brown)

<!--------------------- ETIQUETAS DE ÁREA DEL PROYECTO -------------------------->
[![Electrónica](https://img.shields.io/badge/ÁREA-INTELIGENCIA_ARTIFICIAL-007EC6)](https://github.com/search?q=topic:ad-ia) [![Software](https://img.shields.io/badge/SOFTWARE-E05D44)](https://github.com/search?q=topic:at-software) [![Área](https://img.shields.io/badge/VISIÓN_ARTIFICIAL-701570)](https://github.com/search?q=topic:at-visioncomputadora) 

El siguiente repositorio es un intento de resolver el reto 1 del hackatón "HackICN 2025".

<img width="669" height="836" alt="image" src="https://github.com/user-attachments/assets/3a4aa3a6-2f86-4253-b44a-e68f6fe959bf" />

## Contenido.

# [SURYA IBM/NASA](https://github.com/Additrejo/SolarEx/tree/main/Surya)
<img width="890" height="332" alt="image" src="https://github.com/user-attachments/assets/fdaca304-3564-49f2-bde5-56c7ab955e32" />

El primer modelo fundacional de IA de heliofísica fue entrenado con datos de observación solar de alta resolución. Ofrece insights sobre la superficie dinámica del Sol para ayudar a planificar el clima solar que puede alterar tanto la tecnología en la Tierra como en el espacio.

Visita la carpeta [Surya](https://github.com/Additrejo/SolarEx/tree/main/Surya) para er el despliegue del modelo Surya de IBM/NASA.


# [SolarEx](https://github.com/Additrejo/SolarEx/tree/main/HackICN)
![Solarflex](https://github.com/user-attachments/assets/baba6234-7c8d-4424-a647-89d4264875c7)

Mi primer modelo predictivo (ML) para la identificación de llamaradas solares.
El modelo se basa en la resolución de los puntos solicitados en el reto 1 del hackatón "HackICN 2025".

Este proyecto es un pipeline completo para la detección, análisis y predicción de llamaradas solares a partir de imágenes y videos. 🛰️

Utilizando OpenCV, el sistema primero procesa las imágenes para identificar las llamaradas. Esto se logra mediante la conversión a escala de grises, la aplicación de umbrales (thresholding) para aislar las regiones más brillantes y el análisis de contornos para extraer características clave como el área y la posición (centroide) de cada evento.

Luego, el proyecto entra en una fase de análisis de datos. Al procesar secuencias de video fotograma a fotograma, genera una serie temporal que muestra cómo evoluciona el área de la llamarada. Esta serie de datos se suaviza usando una media móvil con Pandas para eliminar el ruido y se visualiza con Matplotlib.

Finalmente, el proyecto utiliza Machine Learning (específicamente, Regresión Lineal con Scikit-learn) para entrenar un modelo que aprende los patrones de crecimiento de las llamaradas. El resultado es un sistema de alerta temprana capaz de predecir si una llamarada superará un umbral de peligrosidad en los próximos fotogramas, basándose en su comportamiento inicial. El sistema culmina en un script de simulación que procesa un video en tiempo real, dibuja recuadros sobre las llamaradas detectadas y muestra alertas predictivas en pantalla.



## Autores ✒️

* **IBM | NASA** - [SURYA IBM/NASA](https://research.ibm.com/blog/surya-heliophysics-ai-model-sun)
* **Addi Trejo** - *Desarrollador de proyecto* - [additrejo](https://github.com/additrejo)
