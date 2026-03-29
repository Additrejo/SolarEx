# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 01:23:35 2025

@author: addi_
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd # <-- ¡NUEVA IMPORTACIÓN!

def analizar_imagen(ruta_imagen, area_minima=500, umbral=170):
    """
    Analiza una imagen y devuelve el ÁREA MÁXIMA de la llamarada encontrada.
    """
    img = cv2.imread(ruta_imagen)
    if img is None:
        return 0 # Devolvemos 0 si no hay imagen

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    ret, mascara = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY)
    
    contornos, jerarquia = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    area_maxima = 0
    for c in contornos:
        area = cv2.contourArea(c)
        if area > area_minima:
            if area > area_maxima:
                area_maxima = area
                
    return area_maxima

# --- BUCLE PRINCIPAL DE PROCESAMIENTO ---

# 1. Apunta a la carpeta donde guardaste los fotogramas
ruta_carpeta = r"C:/Users/addi_/Downloads/HackICN/Solarflare/SFVideo"

# 2. Parámetros de Alerta
umbral_deteccion = 500
umbral_crecimiento = 2.0 

# 3. Variables para guardar el estado y los datos
area_anterior = 0
alerta_activa = False
fotogramas = []
areas_detectadas = []
puntos_alerta_x = []
puntos_alerta_y = []

if not os.path.isdir(ruta_carpeta):
    print(f" ERROR: La carpeta especificada no existe: {ruta_carpeta}")
else:
    print(f"Iniciando monitoreo de alertas en: {ruta_carpeta}\n")
    lista_archivos = sorted(os.listdir(ruta_carpeta))
    
    # Mismo bucle que antes para recopilar los datos
    for contador_fotograma, nombre_archivo in enumerate(lista_archivos):
        if nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
            area_actual = analizar_imagen(ruta_completa)
            
            if (area_actual > umbral_deteccion and 
                area_anterior > 0 and 
                area_actual > (area_anterior * umbral_crecimiento) and 
                not alerta_activa):
                
                print(f"\n ¡¡ALERTA DE LLAMARADA!!  -> Fotograma: {nombre_archivo}")
                alerta_activa = True
                puntos_alerta_x.append(contador_fotograma)
                puntos_alerta_y.append(area_actual)

            elif area_actual < umbral_deteccion and alerta_activa:
                print(f"-> Evento finalizado en fotograma {contador_fotograma}.")
                alerta_activa = False
            
            fotogramas.append(contador_fotograma)
            areas_detectadas.append(area_actual)
            area_anterior = area_actual

    print(f"\n Análisis completado. Se procesaron {len(fotogramas)} fotogramas.")

    # 6. --- ¡NUEVO! CÁLCULO DE LA MEDIA MÓVIL ---
    if fotogramas:
        print("Calculando media móvil para suavizar la curva...")
        # Define el tamaño de la "ventana" para promediar (ej. 3 fotogramas)
        # ¡Puedes experimentar cambiando este número!
        tamano_ventana = 3
        
        # Convertimos nuestra lista de áreas en un objeto de Pandas (Series)
        series_areas = pd.Series(areas_detectadas)
        
        # Calculamos la media móvil
        # min_periods=1 asegura que tengamos valores desde el principio
        areas_suavizadas = series_areas.rolling(window=tamano_ventana, min_periods=1).mean()

        # 7. Graficar los resultados (con ambas líneas)
        print("Generando gráfica de evolución con media móvil...")
        plt.figure(figsize=(12, 6))
        
        # Dibujar la línea original ruidosa (con transparencia)
        plt.plot(fotogramas, areas_detectadas, label='Área Original (Ruidosa)', alpha=0.4, color='blue')
        
        # Dibujar la nueva línea suavizada
        plt.plot(fotogramas, areas_suavizadas, 
                 label=f'Media Móvil (Ventana={tamano_ventana})', 
                 color='orange', 
                 linewidth=2) # Línea más gruesa y de otro color
        
        # Dibujar los puntos de alerta (aún basados en los datos originales)
        plt.scatter(puntos_alerta_x, puntos_alerta_y, 
                    color='red', 
                    s=100, 
                    zorder=5, 
                    label='¡Alerta de Crecimiento Rápido!')
        
        plt.title('Evolución (Original vs. Suavizada) y Alertas')
        plt.xlabel('Número de Fotograma (Tiempo)')
        plt.ylabel('Área Detectada (en píxeles)')
        plt.grid(True)
        plt.legend()
        plt.show() # Mostrará la gráfica en la pestaña "Plots"
    else:
        print("No se procesaron fotogramas, no se puede graficar.")