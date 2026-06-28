#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
CÁLCULO DE RENDIMIENTOS Y MATRIZ DE CORRELACIONES
El script estudia cómo de correlacionadas están las acciones dentro de sus
respectivos índices y a nivel global (Universo completo).
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# ==============================================================================
# 0. INICIO DEL CRONÓMETRO Y CARPETAS
# ==============================================================================
tiempo_inicio = time.time()
print("Iniciando análisis de correlaciones...")

# Crear el directorio principal para guardar los resultados
dir_corr = "correlaciones/IBEX"
os.makedirs(dir_corr, exist_ok=True)
print(f"Directorio verificado/creado: ./{dir_corr}/")

# ==============================================================================
# 1. CÁLCULO DE RENDIMIENTOS
# ==============================================================================
print("1. Cargando datos...")
# Cargar el dataframe (Asegúrate de que el nombre coincide con tu archivo limpio final)
df = pd.read_csv('cotiz_IBEX_completo.csv', parse_dates=['Date'])

print("2. Calculando rendimientos diarios...")
# Pivotar la tabla: Filas = Fechas, Columnas = Tickers, Valores = Precio de Cierre
df_close = df.pivot(index='Date', columns='Ticker', values='Close')

# Calcular el rendimiento porcentual diario y eliminar la primera fila (que será NaN)[cite: 3]
df_returns = df_close.pct_change().dropna(how='all')

# Crear un mapa rápido para saber a qué índice pertenece cada Ticker[cite: 3]
# Hacemos un pequeño chequeo por si en tu tabla la I de 'Indice' está en mayúscula
col_indice = 'Indice' if 'Indice' in df.columns else 'indice'
df_indices = df[['Ticker', col_indice]].drop_duplicates().set_index('Ticker')

# ==============================================================================
# 2. DEFINICIÓN DE LA FUNCIÓN DE GRAFICADO Y EXPORTACIÓN
# ==============================================================================
def plot_and_save_correlations(df_corr, title, filename_png, filename_csv, is_large=False):
    # 1. Guardar los datos puros en formato .csv
    df_corr.to_csv(filename_csv)
    
    # 2. Generar y guardar el gráfico (Heatmap)
    figsize = (22, 20) if is_large else (14, 12)
    plt.figure(figsize=figsize)
    
    # Máscara para ocultar el triángulo superior (espejo)
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    cmap = sns.color_palette("vlag", as_cmap=True)
    
    # Heatmap
    sns.heatmap(df_corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                annot=not is_large, fmt=".2f", annot_kws={"size": 8},
                square=True, linewidths=.5, cbar_kws={"shrink": .7})
    
    plt.title(title, fontsize=18, pad=20)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    plt.close() # Cerramos el plot para liberar memoria RAM

# ==============================================================================
# 3. EJECUCIÓN POR ÍNDICE Y GLOBAL
# ==============================================================================
print("3. Generando Heatmaps y CSVs por Índice...")
indices_unicos = df_indices[col_indice].dropna().unique()

for indice in indices_unicos:
    print(f"   -> Procesando: {indice}...")
    
    # Sacamos los Tickers que pertenecen a este índice[cite: 3]
    tickers_indice = df_indices[df_indices[col_indice] == indice].index.tolist()
    
    # Filtramos la tabla de retornos cruzando con los tickers que realmente existen
    tickers_validos = [t for t in tickers_indice if t in df_returns.columns]
    retornos_indice = df_returns[tickers_validos]
    
    # Matriz de correlación[cite: 3]
    corr_matrix = retornos_indice.corr()
    
    # Nombres de los archivos
    nombre_limpio = indice.replace(' ', '_')
    ruta_png = os.path.join(dir_corr, f"heatmap_{nombre_limpio}.png")
    ruta_csv = os.path.join(dir_corr, f"correlaciones_{nombre_limpio}.csv")
    
    # Llamamos a la función
    plot_and_save_correlations(corr_matrix, f'Matriz de Correlación - {indice}', ruta_png, ruta_csv)


print("4. Generando Heatmap y CSV GLOBAL (Universo completo)...")
corr_matrix_global = df_returns.corr()
ruta_png_global = os.path.join(dir_corr, "heatmap_GLOBAL.png")
ruta_csv_global = os.path.join(dir_corr, "correlaciones_GLOBAL.csv")

plot_and_save_correlations(corr_matrix_global, 'Matriz de Correlación - TODAS LAS ACCIONES', 
                           ruta_png_global, ruta_csv_global, is_large=True)

print(f"Todos los archivos se han guardado en la carpeta '{dir_corr}'")

# ==============================================================================
# 4. FIN DEL CRONÓMETRO
# ==============================================================================
tiempo_fin = time.time()
tiempo_total = tiempo_fin - tiempo_inicio

horas = int(tiempo_total // 3600)
minutos = int((tiempo_total % 3600) // 60)
segundos = int(tiempo_total % 60)

print("-" * 50)
print(f"El script de correlaciones ha tardado: {horas} horas, {minutos} minutos y {segundos} segundos.")
print("-" * 50)