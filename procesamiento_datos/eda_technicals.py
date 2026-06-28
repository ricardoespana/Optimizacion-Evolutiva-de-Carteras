#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Configuración de carpetas
folder_name = "reporte_eda/tecnicos"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Carpeta '{folder_name}' creada.")

# --- CARGA DE DATOS (Basado en get_technicals.py) ---
archivo = "cotiz_IBEX_completo.csv"
if not os.path.exists(archivo):
    print(f"Error: No se encuentra {archivo}. Ejecuta primero get_technicals.py")
else:
    df = pd.read_csv(archivo)
    df['Date'] = pd.to_datetime(df['Date'])
    df_clean = df.dropna()

    # Columnas de indicadores (Extraídas de[cite: 1])
    cols_indicadores = [
        'RSI_14', 'Stoch_k', 'Stoch_d', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'ADX_14', 'SMA_50', 'SMA_200', 'SMA_20', 'BB_Upper', 'BB_Lower', 
        'ATR_14', 'OBV'
    ]

    # 2. Matriz de Correlación
    plt.figure(figsize=(14, 10))
    correlation_matrix = df_clean[cols_indicadores].corr()
    sns.heatmap(correlation_matrix,
                vmin=-1.0,
                vmax=1.0,
                annot=True,
                cmap='RdBu',
                fmt=".2f")
    plt.title("Matriz de Correlación de Indicadores Técnicos")
    
    # GUARDAR IMAGEN
    path_corr = os.path.join(folder_name, "01_matriz_correlacion.png")
    plt.savefig(path_corr, dpi=300, bbox_inches='tight')
    print(f"Guardado: {path_corr}")
    plt.close() # Cerramos para liberar memoria

    # 3. Histogramas de Distribución
    df_clean[cols_indicadores].hist(bins=50, figsize=(20, 15), color='skyblue', edgecolor='black')
    plt.suptitle("Distribución de los Indicadores Técnicos", fontsize=16)
    
    # GUARDAR IMAGEN
    path_hist = os.path.join(folder_name, "02_distribucion_indicadores.png")
    plt.savefig(path_hist, dpi=300, bbox_inches='tight')
    print(f"Guardado: {path_hist}")
    plt.close()

# 1. Quitamos el .tail() para ver desde 2021

ticker_ejemplo = "A3M.MC"
df_ejemplo = df_clean[df_clean['Ticker'] == ticker_ejemplo] 

# 2. Creamos un gráfico con 2 filas (Precio arriba, RSI abajo)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# GRÁFICO SUPERIOR: Precio + Medias (Escala de Euros)
ax1.plot(df_ejemplo['Date'], df_ejemplo['Close'], label='Precio', color='black')
ax1.plot(df_ejemplo['Date'], df_ejemplo['SMA_50'], label='SMA 50')
ax1.plot(df_ejemplo['Date'], df_ejemplo['SMA_200'], label='SMA 200')
ax1.fill_between(df_ejemplo['Date'], df_ejemplo['BB_Lower'], df_ejemplo['BB_Upper'], color='gray', alpha=0.1)
ax1.set_title("Histórico Completo A3M.MC (2021-2026)")
ax1.legend()

# GRÁFICO INFERIOR: RSI (Escala 0-100)
ax2.plot(df_ejemplo['Date'], df_ejemplo['RSI_14'], color='purple', label='RSI')
ax2.axhline(70, color='red', linestyle='--', alpha=0.5) # Sobrecompra
ax2.axhline(30, color='green', linestyle='--', alpha=0.5) # Sobreventa
ax2.set_ylim(0, 100)
ax2.legend()

  
# GUARDAR IMAGEN
path_ticker = os.path.join(folder_name, f"03_analisis_{ticker_ejemplo}.png")
plt.savefig(path_ticker, dpi=300, bbox_inches='tight')
print(f"Guardado: {path_ticker}")
plt.close()

print("\nPROCESO COMPLETADO: Todas las imágenes están en la carpeta", folder_name)