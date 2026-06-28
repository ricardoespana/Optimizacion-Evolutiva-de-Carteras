#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
PREPROCESAMIENTO FINAL PARA ALGORITMO GENÉTICO (GA) - CORREGIDO
================================================================================
Este script toma el dataset crudo de entrenamiento, aplica la poda de variables
basada en la importancia SHAP, y estandariza los valores numéricos (Z-Score).
Se aplica estandarización libre de look-ahead bias, ajustando el escalador
exclusivamente con los datos del periodo de entrenamiento.
================================================================================
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. PARÁMETROS DE CONFIGURACIÓN
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    ARCHIVO_ENTRADA = os.path.join(BASE_DIR, "GA_Env/data/dataset_S&P500_completo.csv")
    DIR_SALIDA = os.path.join(BASE_DIR, "GA_Env") 
    
    ARCHIVO_SALIDA = os.path.join(DIR_SALIDA, "dataset_GA_estandarizado_S&P_tecnicos.csv")
    ARCHIVO_ESCALADOR = os.path.join(DIR_SALIDA, "scaler_ga_S&P_tecnicos.pkl")
    
    # ── BARRERA TEMPORAL DE ENTRENAMIENTO ──
    # Es vital definir el periodo in-sample para evitar sesgos de futuro
    ANIO_INICIO_TRAIN = 2021
    ANIO_FIN_TRAIN = 2023
    
    # ── ARQUITECTURA DEL CROMOSOMA Y FITNESS ──
    COLS_IDENTIFICADORAS = ['Date', 'Ticker']
    COLS_FITNESS = ['Close', 'Rentabilidad_Diaria']
    
    GENES_ESTADO = [
        # Fundamentales
        #'PER', 
        #'Capitalizacion_Bursatil', 
        #'Rentabilidad_Dividendo',
        #'ROE',
        #'Price-to-book value',
        #'Operating_Margin',
        #'Total_Debt_To_EBITDA',
        #'Ratio_Tier_1',
        #'Solvency_Ratio',
        #'Combined_Ratio',
        #'Margen_EBITDA',
        # Técnicos y NLP
        'OBV', 
        'SMA_200', 
        'SMA_50', 
        'ATR_14', 
        'MACD_Signal',
        'Sentimiento_Num',
        'Volatilidad_Diaria'
    ]

# Crear directorio de salida
os.makedirs(Config.DIR_SALIDA, exist_ok=True)

# ==============================================================================
# 2. CARGA Y PODA DE DATOS
# ==============================================================================
print("="*70)
print("⚙️ INICIANDO PIPELINE DE PREPROCESAMIENTO PARA ALGORITMO GENÉTICO")
print("="*70)

df_completo = pd.read_csv(Config.ARCHIVO_ENTRADA)

if 'P/E' in df_completo.columns and 'PER' not in df_completo.columns:
    df_completo.rename(columns={'P/E': 'PER'}, inplace=True)

cols_faltantes = [c for c in Config.GENES_ESTADO if c not in df_completo.columns]
if cols_faltantes:
    print(f"⚠️ ADVERTENCIA: Faltan estas columnas (genes) y serán ignoradas: {cols_faltantes}")
    Config.GENES_ESTADO = [c for c in Config.GENES_ESTADO if c in df_completo.columns]

columnas_finales = Config.COLS_IDENTIFICADORAS + Config.COLS_FITNESS + Config.GENES_ESTADO
df_optimizado = df_completo[columnas_finales].copy()

print(f"   -> Tamaño del cromosoma reducido a {len(Config.GENES_ESTADO)} genes.")

# ==============================================================================
# 3. ESTANDARIZACIÓN ESTRICTA (ANTI LOOK-AHEAD BIAS)
# ==============================================================================
print(f"2. Ajustando escalador SOLO con datos de {Config.ANIO_INICIO_TRAIN} a {Config.ANIO_FIN_TRAIN}...")

df_optimizado['Date'] = pd.to_datetime(df_optimizado['Date'])
df_optimizado = df_optimizado.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

# 1. Aislamos el periodo de entrenamiento
mask_train = (df_optimizado['Date'].dt.year >= Config.ANIO_INICIO_TRAIN) & (df_optimizado['Date'].dt.year <= Config.ANIO_FIN_TRAIN)
df_train = df_optimizado[mask_train]

if df_train.empty:
    raise ValueError(f"No hay datos en el periodo de entrenamiento {Config.ANIO_INICIO_TRAIN}-{Config.ANIO_FIN_TRAIN}.")

# 2. Instanciamos el escalador
scaler = StandardScaler()

# 3. FIT: El escalador aprende las matemáticas exclusivas del pasado
scaler.fit(df_train[Config.GENES_ESTADO])

# 4. TRANSFORM: Se aplican esas métricas del pasado a toda la matriz (Train y Backtest)
df_optimizado[Config.GENES_ESTADO] = scaler.transform(df_optimizado[Config.GENES_ESTADO])

# ==============================================================================
# 4. GUARDADO DE ARTEFACTOS
# ==============================================================================
print("3. Exportando dataset para la población inicial...")

df_optimizado.to_csv(Config.ARCHIVO_SALIDA, index=False)
joblib.dump(scaler, Config.ARCHIVO_ESCALADOR)

print(f"   -> Dataset GA guardado en: {Config.ARCHIVO_SALIDA}")
print(f"   -> Escalador guardado en:  {Config.ARCHIVO_ESCALADOR}")

print("\n" + "="*70)
print("✅ PREPROCESAMIENTO SEGURO FINALIZADO. DATOS LISTOS PARA LA EVOLUCIÓN.")