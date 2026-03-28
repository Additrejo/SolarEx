# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 00:05:49 2025

@author: addi_
"""

import cv2
import numpy as np

# --- ¡MODIFICA ESTAS LÍNEAS! ---
# Coloca la ruta COMPLETA y CORRECTA de tu imagen aquí.
ruta_de_tu_imagen = r"C:/Users/addi_/Downloads/HackICN/Solarflare/SDO-solarflare.jpg"
# Define el área mínima en píxeles para considerar una detección.
# ¡Experimenta con este valor!
area_minima = 500

try:
    # 1. Cargar la imagen
    img = cv2.imread(ruta_de_tu_imagen)
    if img is None:
        print(f" ERROR: No se pudo cargar la imagen. Verifica la ruta:\n{ruta_de_tu_imagen}")
    else:
        print(" ¡Imagen cargada con éxito!")
        output_img = img.copy()
        
        # 2. Preprocesamiento
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # 3. Umbralización
        # Ajusta el valor 200 según la intensidad de la llamarada en tu imagen.
        ret, mascara = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        
        # 4. Encontrar los contornos en la máscara
        contornos, jerarquia = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detecciones = 0
        # 5. --- ¡NUEVO! --- Bucle para analizar cada contorno
        for c in contornos:
            # Calcular el área del contorno actual
            area = cv2.contourArea(c)
            
            # Si el área es mayor que nuestro mínimo, lo procesamos
            if area > area_minima:
                detecciones += 1
                # Dibujar solo el contorno que pasó el filtro
                cv2.drawContours(output_img, [c], -1, (0, 255, 0), 2)
                print(f"Detección {detecciones}: Contorno encontrado con área de {int(area)} píxeles.")

        if detecciones == 0:
            print(f"No se encontraron contornos con área mayor a {area_minima} píxeles.")
        else:
             print(f"Total de {detecciones} detecciones válidas encontradas.")

        # 6. Mostrar los resultados
        cv2.imshow("Imagen Original", img)
        cv2.imshow("Mascara (Resultado del Umbral)", mascara)
        cv2.imshow("Detecciones Filtradas por Area", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")