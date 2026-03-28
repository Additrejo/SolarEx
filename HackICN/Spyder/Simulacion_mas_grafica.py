# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 00:54:58 2025

@author: addi_
"""


import cv2
import numpy as np
import pandas as pd
import joblib
from collections import deque

# --- PARÁMETROS GLOBALES ---
AREA_MINIMA = 500
UMBRAL_CV = 170
TAMANO_VENTANA_MOVIL = 3
TAMANO_VENTANA_ML = 5
UMBRAL_ALERTA_ML = 2000
ANCHO_PANTALLA_DESEADO = 960 # Antes 1280

# --- ¡NUEVO! Parámetros para la predicción en línea ---
# Después de este número de fotogramas, empezaremos a mostrar la línea de predicción
# ajusta esto según cuándo quieras que el modelo "tome el control" visualmente
FOTOGRAMA_INICIO_PREDICCION_LINEA = 80 
NUM_PASOS_PREDICCION_FUTURA = 20 # Cuántos puntos futuros queremos predecir y dibujar

# --- Función de Detección (la misma de antes) ---
def analizar_imagen_con_recuadro(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    ret, mascara = cv2.threshold(gray, UMBRAL_CV, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_maxima = 0
    recuadro = None
    if len(contornos) > 0:
        c_max = max(contornos, key=cv2.contourArea)
        area = cv2.contourArea(c_max)
        if area > AREA_MINIMA:
            area_maxima = area
            recuadro = cv2.boundingRect(c_max)
    return area_maxima, recuadro

# --- Función para DIBUJAR LA GRÁFICA ---
def dibujar_grafica_mejorada(frame_grafica, historial_datos_raw, historial_datos_suavizados, 
                             historial_predicciones_linea, # <-- ¡NUEVO!
                             max_area_vista, puntos_de_alerta, fotograma_actual,
                             area_real, prediccion_val, es_alerta): 
    h, w, _ = frame_grafica.shape
    frame_grafica.fill(255) # Fondo blanco
    
    color_grid = (220, 220, 220)
    color_texto = (0, 0, 0)
    color_linea_raw = (200, 200, 0) # Amarillo claro
    color_linea_suavizada = (0, 165, 255) # Azul/cian (Media móvil)
    color_linea_prediccion = (0, 0, 255) # Rojo (Línea punteada) <-- ¡NUEVO!
    color_alerta_tendencia = (0, 0, 255) # Rojo (Puntos de alerta)
    color_linea_inicio_pred = (0, 150, 0) # Verde para "Inicio de Pruebas" <-- ¡NUEVO!
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_ejes = 0.5
    font_scale_leyenda = 0.4
    thickness = 1
    
    # --- 1. Definir Márgenes y Áreas ---
    margen_sup = 60  
    margen_inf = 80 
    margen_izq = 80 
    margen_der = 180 
    
    plot_y_inicio = margen_sup
    plot_h = h - margen_sup - margen_inf 
    plot_y_fin = plot_y_inicio + plot_h
    plot_x_inicio = margen_izq
    plot_w = w - margen_izq - margen_der 
    plot_x_fin = plot_x_inicio + plot_w
    
    # --- 2. Título (Arriba, centrado) ---
    titulo = "Deteccion de Alertas Robusta (Basada en Media Movil)"
    (text_w, text_h), _ = cv2.getTextSize(titulo, font, 0.7, thickness)
    cv2.putText(frame_grafica, titulo, (w // 2 - text_w // 2, 30),
                font, 0.7, color_texto, thickness)

    # --- 3. Etiquetas de Ejes (Títulos) ---
    # Eje Y (Rotado, a la izquierda del plot)
    titulo_y = "Area Detectada (en pixeles)"
    (text_w_y, text_h_y), _ = cv2.getTextSize(titulo_y, font, font_scale_ejes, thickness)
    temp_img_y = np.zeros((text_w_y + 10, text_h_y + 10, 3), dtype=np.uint8) + 255
    cv2.putText(temp_img_y, titulo_y, (5, text_w_y + 5), font, font_scale_ejes, color_texto, thickness)
    temp_img_y = cv2.rotate(temp_img_y, cv2.ROTATE_90_COUNTERCLOCKWISE)
    y_pos_y_label = plot_y_inicio + plot_h // 2 - text_h_y // 2
    x_pos_y_label = 10 
    end_x_y_label = x_pos_y_label + temp_img_y.shape[1]
    end_y_y_label = y_pos_y_label + temp_img_y.shape[0]
    if y_pos_y_label < 0: y_pos_y_label = 0
    if x_pos_y_label < 0: x_pos_y_label = 0
    if end_x_y_label > w: end_x_y_label = w
    if end_y_y_label > h: end_y_y_label = h
    frame_grafica[y_pos_y_label : end_y_y_label, x_pos_y_label : end_x_y_label] = \
        temp_img_y[0 : end_y_y_label - y_pos_y_label, 0 : end_x_y_label - x_pos_y_label]

    # Eje X (Debajo del plot, centrado)
    titulo_x = "Numero de fotograma (Tiempo)"
    (text_w_x, text_h_x), _ = cv2.getTextSize(titulo_x, font, font_scale_ejes, thickness)
    cv2.putText(frame_grafica, titulo_x, (plot_x_inicio + plot_w // 2 - text_w_x // 2, plot_y_fin + 60),
                font, font_scale_ejes, color_texto, thickness)

    # --- 4. Escalas Numéricas en Ejes ---
    max_area_display = max(max_area_vista, UMBRAL_ALERTA_ML * 1.5) 
    num_y_ticks = 5 
    for i in range(num_y_ticks + 1):
        value = int(max_area_display * (i / num_y_ticks))
        y_coord = int(plot_y_fin - (value / max_area_display) * plot_h)
        cv2.line(frame_grafica, (plot_x_inicio - 5, y_coord), (plot_x_inicio, y_coord), color_texto, 1) 
        cv2.putText(frame_grafica, str(value), (plot_x_inicio - 60, y_coord + 5), font, font_scale_ejes, color_texto, thickness)

    num_x_ticks = 5 
    for i in range(num_x_ticks + 1):
        value = int((fotograma_actual / num_x_ticks) * i) # Escala hasta el fotograma actual
        if value > fotograma_actual and i != num_x_ticks: continue 
        x_coord = int(plot_x_inicio + (value / fotograma_actual) * plot_w) if fotograma_actual > 0 else plot_x_inicio
        cv2.line(frame_grafica, (x_coord, plot_y_fin), (x_coord, plot_y_fin + 5), color_texto, 1) 
        cv2.putText(frame_grafica, str(value), (x_coord - 15, plot_y_fin + 20), font, font_scale_ejes, color_texto, thickness)

    # --- 5. Leyenda (Arriba, derecha) ---
    legend_x_start = plot_x_fin + 10
    legend_y_start = plot_y_inicio
    line_height = 25
    
    cv2.line(frame_grafica, (legend_x_start, legend_y_start), (legend_x_start + 40, legend_y_start), 
             color_linea_raw, thickness, lineType=cv2.LINE_AA)
    cv2.putText(frame_grafica, "Area Original (Ruidosa)", (legend_x_start + 50, legend_y_start + 5), 
                font, font_scale_leyenda, color_texto, thickness)
    
    cv2.line(frame_grafica, (legend_x_start, legend_y_start + line_height), (legend_x_start + 40, legend_y_start + line_height), 
             color_linea_suavizada, 2, lineType=cv2.LINE_AA)
    cv2.putText(frame_grafica, f"Media Movil (Ventana={TAMANO_VENTANA_MOVIL})", (legend_x_start + 50, legend_y_start + line_height + 5), 
                font, font_scale_leyenda, color_texto, thickness)
    
    # Leyenda para la línea de predicción
    cv2.line(frame_grafica, (legend_x_start, legend_y_start + 2 * line_height), (legend_x_start + 40, legend_y_start + 2 * line_height), 
             color_linea_prediccion, thickness, lineType=cv2.LINE_AA) # No se puede hacer punteada directamente
    cv2.putText(frame_grafica, "Predicciones del Modelo (ML)", (legend_x_start + 50, legend_y_start + 2 * line_height + 5), 
                font, font_scale_leyenda, color_texto, thickness)
    
    cv2.circle(frame_grafica, (legend_x_start + 20, legend_y_start + 3 * line_height), 5, 
               color_alerta_tendencia, -1, lineType=cv2.LINE_AA)
    cv2.putText(frame_grafica, "¡Alerta de tendencia!", (legend_x_start + 50, legend_y_start + 3 * line_height + 5), 
                font, font_scale_leyenda, color_texto, thickness)
    
    # Leyenda para el inicio de pruebas
    cv2.line(frame_grafica, (legend_x_start, legend_y_start + 4 * line_height), (legend_x_start + 40, legend_y_start + 4 * line_height), 
             color_linea_inicio_pred, thickness, lineType=cv2.LINE_AA)
    cv2.putText(frame_grafica, "Inicio de Pruebas", (legend_x_start + 50, legend_y_start + 4 * line_height + 5), 
                font, font_scale_leyenda, color_texto, thickness)


    # --- 6. Cuadrícula (Solo en el área del plot) ---
    for i in range(num_y_ticks + 1): 
        y_grid = int(plot_y_inicio + plot_h * (i / num_y_ticks))
        cv2.line(frame_grafica, (plot_x_inicio, y_grid), (plot_x_fin, y_grid), color_grid, 1)
    
    for i in range(num_x_ticks + 1): 
        x_grid = int(plot_x_inicio + plot_w * (i / num_x_ticks))
        cv2.line(frame_grafica, (x_grid, plot_y_inicio), (x_grid, plot_y_fin), color_grid, 1) 

    # --- 7. Dibujo de Líneas y Alertas ---
    num_puntos_historial = len(historial_datos_suavizados)
    if num_puntos_historial < 2:
        return frame_grafica 

    max_val_plot = max(max_area_vista, UMBRAL_ALERTA_ML * 1.2, max(historial_predicciones_linea + historial_datos_suavizados) if historial_predicciones_linea else 0) 
    
    # Puntos para la línea suavizada (datos reales)
    puntos_suavizados = []
    for i in range(num_puntos_historial):
        x = int(plot_x_inicio + i * (plot_w / fotograma_actual)) if fotograma_actual > 0 else plot_x_inicio
        y_suav = int(plot_y_fin - (historial_datos_suavizados[i] / max_val_plot) * plot_h)
        puntos_suavizados.append([x, y_suav])

    cv2.polylines(frame_grafica, [np.array(puntos_suavizados)], isClosed=False, 
                  color=color_linea_suavizada, thickness=2, lineType=cv2.LINE_AA)
    
    # Puntos para la línea raw (ruidosa)
    puntos_raw = []
    for i in range(num_puntos_historial):
        x = int(plot_x_inicio + i * (plot_w / fotograma_actual)) if fotograma_actual > 0 else plot_x_inicio
        y_raw = int(plot_y_fin - (historial_datos_raw[i] / max_val_plot) * plot_h) 
        puntos_raw.append([x, y_raw])
    cv2.polylines(frame_grafica, [np.array(puntos_raw)], isClosed=False, 
                  color=color_linea_raw, thickness=thickness, lineType=cv2.LINE_AA)

    # Dibujar línea vertical de "Inicio de Pruebas"
    if fotograma_actual > FOTOGRAMA_INICIO_PREDICCION_LINEA:
        x_inicio_pred = int(plot_x_inicio + (FOTOGRAMA_INICIO_PREDICCION_LINEA / fotograma_actual) * plot_w)
        # Dibujar línea punteada
        for y_step in range(plot_y_inicio, plot_y_fin, 10): # Puntos cada 10 píxeles
            cv2.line(frame_grafica, (x_inicio_pred, y_step), (x_inicio_pred, y_step + 5), color_linea_inicio_pred, 1)


    # ¡NUEVO! Dibujar la línea de predicción punteada
    puntos_prediccion_linea = []
    if historial_predicciones_linea:
        # El primer punto de la predicción debe unirse con el último dato real suavizado
        last_real_x = puntos_suavizados[-1][0] if puntos_suavizados else plot_x_inicio
        last_real_y = puntos_suavizados[-1][1] if puntos_suavizados else plot_y_fin
        
        # Solo dibujamos la predicción si hemos superado el punto de inicio
        if fotograma_actual >= FOTOGRAMA_INICIO_PREDICCION_LINEA:
            # Añadir el último punto real suavizado como punto de partida para la predicción
            if puntos_suavizados:
                puntos_prediccion_linea.append(puntos_suavizados[-1])
            
            for i, val_pred in enumerate(historial_predicciones_linea):
                # Escalamos en base al fotograma actual + índice de la predicción
                x = int(plot_x_inicio + (fotograma_actual + i) * (plot_w / (fotograma_actual + NUM_PASOS_PREDICCION_FUTURA))) # Escalar X
                y = int(plot_y_fin - (val_pred / max_val_plot) * plot_h)
                puntos_prediccion_linea.append([x, y])
            
            # Dibujar como línea punteada
            for i in range(len(puntos_prediccion_linea) - 1):
                p1 = puntos_prediccion_linea[i]
                p2 = puntos_prediccion_linea[i+1]
                # Implementación simple de línea punteada
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                if dist > 0:
                    num_dashes = int(dist / 10) # Aproximadamente un dash cada 10 pixeles
                    for j in range(num_dashes):
                        start_dash_x = int(p1[0] + (p2[0] - p1[0]) * (j / num_dashes))
                        start_dash_y = int(p1[1] + (p2[1] - p1[1]) * (j / num_dashes))
                        end_dash_x = int(p1[0] + (p2[0] - p1[0]) * ((j + 0.5) / num_dashes))
                        end_dash_y = int(p1[1] + (p2[1] - p1[1]) * ((j + 0.5) / num_dashes))
                        cv2.line(frame_grafica, (start_dash_x, start_dash_y), (end_dash_x, end_dash_y), 
                                 color_linea_prediccion, thickness, lineType=cv2.LINE_AA)

    if len(puntos_suavizados) > 0:
        cv2.line(frame_grafica, (puntos_suavizados[-1][0], plot_y_inicio), (puntos_suavizados[-1][0], plot_y_fin), (100, 100, 100), 1)

    for (fotograma_idx, valor) in puntos_de_alerta:
        idx = fotograma_idx - 1 
        if idx < num_puntos_historial:
            x_alerta = int(plot_x_inicio + idx * (plot_w / fotograma_actual)) if fotograma_actual > 0 else plot_x_inicio
            y_alerta = int(plot_y_fin - (valor / max_val_plot) * plot_h)
            cv2.circle(frame_grafica, (x_alerta, y_alerta), 7, color_alerta_tendencia, -1, lineType=cv2.LINE_AA)
            
    # --- 8. Texto dinámico ahora en el TOP-LEFT del plot ---
    info_texto_y = plot_y_inicio + 30 
    margen_texto_izq = plot_x_inicio + 20
    
    cv2.putText(frame_grafica, f"Area (real): {area_real:.0f}", 
                (margen_texto_izq, info_texto_y),
                font, 0.7, (0, 0, 0), 2)
    cv2.putText(frame_grafica, f"Prediccion: {prediccion_val:.0f}", 
                (margen_texto_izq, info_texto_y + 30),
                font, 0.7, (0, 0, 200), 2)
    if es_alerta:
        texto_alerta_final = "!! ALERTA PREDICTIVA !!"
        cv2.putText(frame_grafica, texto_alerta_final, (margen_texto_izq, info_texto_y + 60), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)

    return frame_grafica

# --- 1. CARGA DEL MODELO Y PREPARACIÓN ---
print("Cargando modelo de IA 'predictor_llamaradas.pkl'...")
try:
    modelo = joblib.load('predictor_llamaradas.pkl')
except FileNotFoundError:
    print(" ERROR: No se encontró el archivo 'predictor_llamaradas.pkl'.")
    exit()

ruta_video_nuevo = r"C:/Users/addi_/Downloads/HackICN/Solarflare/SFVideo/SFVideo.mp4" # ¡MODIFICA ESTO!
video = cv2.VideoCapture(ruta_video_nuevo)

if not video.isOpened():
    print(f" ERROR: No se pudo abrir el video en: {ruta_video_nuevo}")
    exit()

ancho = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

ancho_final = ancho * 2
lienzo_final = np.zeros((alto, ancho_final, 3), dtype=np.uint8)
frame_grafica_der = np.zeros((alto, ancho, 3), dtype=np.uint8) 

video_salida = cv2.VideoWriter('simulacion_lado_a_lado.mp4', 
                               cv2.VideoWriter_fourcc(*'mp4v'), 
                               fps, (ancho_final, alto))

# Historiales completos para la gráfica
historial_completo_raw = []       
historial_completo_suavizado = [] 
historial_puntos_alerta = [] 
max_area_vista = 1 
alerta_activa = False
fotograma_actual = 0
historial_predicciones_futuras = [] # <-- ¡NUEVO! Para almacenar la línea predicha

# --- 2. BUCLE DE SIMULACIÓN EN TIEMPO REAL ---
print("Iniciando simulación... Presiona 'q' para salir.")
while True:
    exito, frame = video.read()
    if not exito:
        break 

    fotograma_actual += 1

    # --- LADO IZQUIERDO: Detección en Video ---
    frame_display_izq = frame.copy() 

    area_actual, recuadro = analizar_imagen_con_recuadro(frame_display_izq)
    if recuadro:
        x, y, w_rec, h_rec = recuadro
        color_deteccion = (0, 0, 255) # Rojo (BGR)
        cv2.rectangle(frame_display_izq, (x, y), (x + w_rec, y + h_rec), color_deteccion, 2)
        cv2.putText(frame_display_izq, f"Area: {int(area_actual)} px", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_deteccion, 2)

    # --- LÓGICA DE PREDICCIÓN CON LÍNEA PREDICHA ---
    historial_completo_raw.append(area_actual) 
    
    series_suavizada_pd = pd.Series(historial_completo_raw).rolling(window=TAMANO_VENTANA_MOVIL, min_periods=1).mean()
    area_suavizada_actual = series_suavizada_pd.iloc[-1]
    historial_completo_suavizado.append(area_suavizada_actual) 
    
    if area_actual > max_area_vista:
        max_area_vista = area_actual
        
    prediccion_valor_actual = 0 # Esta es la predicción puntual (para el texto)
    historial_predicciones_futuras = [] # Reiniciamos para cada fotograma

    # Si tenemos suficientes datos para predecir, y hemos superado el umbral de inicio de predicción visual
    if len(historial_completo_suavizado) >= TAMANO_VENTANA_ML and fotograma_actual >= FOTOGRAMA_INICIO_PREDICCION_LINEA:
        
        # Generar la serie de predicciones futuras
        datos_para_predecir_serie = list(historial_completo_suavizado[-TAMANO_VENTANA_ML:])
        
        # Para la predicción puntual mostrada en el texto
        if not np.isnan(datos_para_predecir_serie).any():
            prediccion_valor_actual = modelo.predict(np.array(datos_para_predecir_serie).reshape(1, -1))[0]
        
        # Generar N pasos de predicción para la línea
        temp_datos_pred = list(datos_para_predecir_serie)
        for _ in range(NUM_PASOS_PREDICCION_FUTURA):
            if len(temp_datos_pred) >= TAMANO_VENTANA_ML and not np.isnan(temp_datos_pred[-TAMANO_VENTANA_ML:]).any():
                next_pred = modelo.predict(np.array(temp_datos_pred[-TAMANO_VENTANA_ML:]).reshape(1, -1))[0]
                historial_predicciones_futuras.append(next_pred)
                temp_datos_pred.append(next_pred) # Añadir la predicción a la ventana para la siguiente
            else:
                break # Si no hay suficientes datos para predecir, detenerse
                
    if prediccion_valor_actual > UMBRAL_ALERTA_ML and not alerta_activa:
        alerta_activa = True
        historial_puntos_alerta.append((fotograma_actual, area_suavizada_actual))
    elif prediccion_valor_actual < UMBRAL_ALERTA_ML and alerta_activa:
        alerta_activa = False

    # --- LADO DERECHO: Dibujo de Gráfica ---
    frame_grafica_der = dibujar_grafica_mejorada(frame_grafica_der, 
                                                historial_completo_raw,       
                                                historial_completo_suavizado, 
                                                historial_predicciones_futuras, # <-- ¡NUEVO!
                                                max_area_vista, 
                                                historial_puntos_alerta,
                                                fotograma_actual,
                                                area_suavizada_actual, 
                                                prediccion_valor_actual,            
                                                alerta_activa)         
    
    if alerta_activa:
        texto_alerta_izq = "!! ALERTA PREDICTIVA !!"
        cv2.putText(frame_display_izq, texto_alerta_izq, (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)
        
    # --- ENSAMBLAJE FINAL ---
    lienzo_final[0:alto, 0:ancho] = frame_display_izq
    lienzo_final[0:alto, ancho:ancho_final] = frame_grafica_der

    # Redimensionar la ventana para mostrar
    relacion_aspecto = alto / ancho_final
    alto_pantalla_deseado = int(ANCHO_PANTALLA_DESEADO * relacion_aspecto)
    
    lienzo_mostrado = cv2.resize(lienzo_final, (ANCHO_PANTALLA_DESEADO, alto_pantalla_deseado), 
                                 interpolation=cv2.INTER_AREA)

    cv2.imshow('Simulacion Lado-a-Lado (Presiona q para salir)', lienzo_mostrado)
    video_salida.write(lienzo_final) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 3. LIMPIEZA ---
video.release()
video_salida.release()
cv2.destroyAllWindows()
print("Simulación finalizada. Video guardado en 'simulacion_lado_a_lado.mp4'.")