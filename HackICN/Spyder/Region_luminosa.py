# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 23:04:30 2025

Este código identificará el punto exacto de mayor intensidad en la imagen y dibujará un círculo sobre él para que podamos visualizarlo.

@author: addi_
"""
import cv2
import numpy as np

# --- ¡MODIFICA ESTA LÍNEA! ---
# Coloca la ruta COMPLETA y CORRECTA de tu imagen aquí.
# Usar una 'r' antes de las comillas ayuda a que Python interprete bien las rutas de Windows.
# ¡No olvides incluir el nombre del archivo y su extensión (ej: .jpg, .png)!
ruta_de_tu_imagen = r"C:/Users/addi_/Downloads/HackICN/Solarflare/SDO-solarflare.jpg"

try:
    # 1. Cargar la imagen directamente desde la ruta local con cv2.imread()
    img = cv2.imread(ruta_de_tu_imagen)

    # 2. Comprobar si la imagen se cargó correctamente
    if img is None:
        print(f" ERROR: No se pudo cargar la imagen.")
        print(f"Verifica que la ruta es correcta y el archivo existe: \n{ruta_de_tu_imagen}")
    else:
        # Si la imagen se cargó, continuamos con el procesamiento
        print(" ¡Imagen cargada con éxito!")
        
        # Guardar una copia a color para dibujar sobre ella
        output_img = img.copy()
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar desenfoque
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Encontrar el píxel más brillante
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        
        # Dibujar un círculo en la ubicación
        cv2.circle(output_img, maxLoc, 40, (0, 0, 255), 2)

        # Mostrar los resultados
        cv2.imshow("Imagen Original", img)
        cv2.imshow("Deteccion de Llamarada", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")