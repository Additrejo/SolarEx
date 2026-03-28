# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 00:38:58 2025

@author: addi_
"""

import cv2
import numpy as np

# --- ¡MODIFICA ESTAS LÍNEAS! ---
# Coloca la ruta COMPLETA y CORRECTA de tu imagen aquí.
ruta_de_tu_imagen = r"C:/Users/addi_/Downloads/HackICN/Solarflare/SDO-solarflare.jpg"
# Define el área mínima en píxeles para considerar una detección.
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
        ret, mascara = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        
        # 4. Encontrar los contornos en la máscara
        contornos, jerarquia = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Bucle para analizar cada contorno
        for i, c in enumerate(contornos):
            area = cv2.contourArea(c)
            
            if area > area_minima:
                # --- ¡NUEVO! --- Calcular el centroide
                # Primero, calculamos los momentos del contorno
                M = cv2.moments(c)
                
                # Evitamos división por cero y calculamos las coordenadas (x, y) del centro
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    # Si el área es cero, usamos un valor por defecto
                    cX, cY = 0, 0
                
                # Dibujar el contorno y el centroide en la imagen de salida
                cv2.drawContours(output_img, [c], -1, (0, 255, 0), 2) # Contorno verde
                cv2.circle(output_img, (cX, cY), 2, (0, 0, 255), -1) # Círculo Rojo en el centro. (B,G,R)
                cv2.putText(output_img, f"({cX}, {cY})", (cX - 40, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) #Letras de origen en color amarillo.

                print(f"Detección {i+1}: Área={int(area)} píxeles, Centroide=({cX}, {cY})")

        # 6. Mostrar los resultados
        cv2.imshow("Imagen Original", img)
        cv2.imshow("Detecciones con Centroide", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")