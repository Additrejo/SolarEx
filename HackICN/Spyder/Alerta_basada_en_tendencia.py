# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 01:31:22 2025

@author: addi_
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

def analizar_imagen(ruta_imagen, area_minima=500, umbral=170):
    """
    Analiza una imagen y devuelve el ÁREA MÁXIMA de la llamarada encontrada.
    """
    img = cv2.imread(ruta_imagen)
    if img is None:
        return 0
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

# --- PARÁMETROS DE PROCESAMIENTO ---
ruta_carpeta = r"C:/Users/addi_/Downloads/HackICN/Solarflare/SFVideo"
tamano_ventana_movil = 3 # Ventana de 3 fotogramas para la media móvil
umbral_deteccion = 500   # Área mínima para empezar a considerar una alerta
umbral_crecimiento = 1.8 # Alerta si el área crece un 80% (1.8)

# --- 1. FASE DE RECOLECCIÓN DE DATOS ---
fotogramas = []
areas_detectadas = [] # Lista para los datos crudos (ruidosos)

if not os.path.isdir(ruta_carpeta):
    print(f" ERROR: La carpeta especificada no existe: {ruta_carpeta}")
else:
    print(f"Iniciando Fase 1: Recolección de datos en: {ruta_carpeta}\n")
    lista_archivos = sorted(os.listdir(ruta_carpeta))
    
    for contador_fotograma, nombre_archivo in enumerate(lista_archivos):
        if nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
            area_actual = analizar_imagen(ruta_completa)
            
            fotogramas.append(contador_fotograma)
            areas_detectadas.append(area_actual)
            
    print(f" Recolección completa. Se procesaron {len(fotogramas)} fotogramas.")

    # --- 2. FASE DE ANÁLISIS Y ALERTA (SOBRE DATOS SUAVIZADOS) ---
    if fotogramas:
        print("Iniciando Fase 2: Cálculo de media móvil y detección de alertas...")
        
        # 2a. Calcular la serie de media móvil
        series_areas = pd.Series(areas_detectadas)
        areas_suavizadas = series_areas.rolling(window=tamano_ventana_movil, min_periods=1).mean()

        # 2b. Buscar alertas en los datos SUAVIZADOS
        alerta_activa = False
        puntos_alerta_x = []
        puntos_alerta_y = []

        # Empezamos desde 1 para poder comparar con el anterior
        for i in range(1, len(areas_suavizadas)):
            area_actual_suavizada = areas_suavizadas[i]
            area_anterior_suavizada = areas_suavizadas[i-1]

            if (area_actual_suavizada > umbral_deteccion and 
                area_anterior_suavizada > 0 and 
                area_actual_suavizada > (area_anterior_suavizada * umbral_crecimiento) and 
                not alerta_activa):
                
                print(f"\n ¡¡ALERTA DE LLAMARADA ROBUSTA!! ")
                print(f"   En el fotograma: Nº {i}")
                print(f"   Crecimiento de tendencia detectado: de {area_anterior_suavizada:.0f} a {area_actual_suavizada:.0f} (promedio).")
                
                alerta_activa = True
                puntos_alerta_x.append(i) # Guardamos el índice (fotograma)
                puntos_alerta_y.append(area_actual_suavizada) # Guardamos el valor de la alerta

            elif area_actual_suavizada < umbral_deteccion and alerta_activa:
                print(f"-> Evento finalizado en fotograma {i}.")
                alerta_activa = False

        # --- 3. FASE DE GRAFICACIÓN ---
        print("\nGenerando gráfica final...")
        plt.figure(figsize=(12, 6))
        
        # Línea original (ruidosa)
        plt.plot(fotogramas, areas_detectadas, label='Área Original (Ruidosa)', alpha=0.3, color='blue')
        
        # Línea suavizada (tendencia)
        plt.plot(fotogramas, areas_suavizadas, 
                 label=f'Media Móvil (Ventana={tamano_ventana_movil})', 
                 color='orange', 
                 linewidth=2.5)
        
        # Puntos de alerta (basados en la línea suavizada)
        plt.scatter(puntos_alerta_x, puntos_alerta_y, 
                    color='red', 
                    s=120, # Marcador más grande
                    zorder=5, 
                    label='¡Alerta de Tendencia!')
        
        plt.title('Detección de Alertas Robusta (Basada en Media Móvil)')
        plt.xlabel('Número de Fotograma (Tiempo)')
        plt.ylabel('Área Detectada (en píxeles)')
        plt.grid(True)
        plt.legend()
        plt.show() # Mostrará la gráfica en la pestaña "Plots"
    else:
        print("No se procesaron fotogramas, no se puede graficar.")