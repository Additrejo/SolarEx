# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 23:36:14 2025

@author: addi_
"""

import cv2
import numpy as np

# --- ¡MODIFICA ESTA LÍNEA! ---
# Coloca la ruta COMPLETA y CORRECTA de tu imagen aquí.
ruta_de_tu_imagen = r"C:/Users/addi_/Downloads/HackICN/Solarflare/SDO-solarflare.jpg"

try:
    # 1. Cargar la imagen
    img = cv2.imread(ruta_de_tu_imagen)
    if img is None:
        print(f" ERROR: No se pudo cargar la imagen. Verifica la ruta:\n{ruta_de_tu_imagen}")
    else:
        print(" ¡Imagen cargada con éxito!")
        output_img = img.copy()
        
        # 2. Preprocesamiento (igual que antes)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # 3. Umbralización (Thresholding)
        # Los píxeles con valor > 200 se convertirán en 255 (blanco).
        # El resto se convertirán en 0 (negro).
        # ¡El valor 200 es el que puedes ajustar!
        ret, mascara = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # 4. Encontrar los contornos en la máscara
        # Esto nos da una lista de todas las formas blancas detectadas.
        contornos, jerarquia = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Se encontraron {len(contornos)} contornos/regiones brillantes.")

        # 5. Dibujar los contornos sobre la imagen original
        # El -1 indica que queremos dibujar TODOS los contornos encontrados.
        cv2.drawContours(output_img, contornos, -1, (0, 255, 0), 2) # Contorno verde

        # 6. Mostrar los resultados
        cv2.imshow("Imagen Original", img)
        cv2.imshow("Mascara (Resultado del Umbral)", mascara)
        cv2.imshow("Deteccion con Contornos", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")