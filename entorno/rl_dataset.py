#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
GENERADOR DE DATASETS: CRUDO Y ENTRENAMIENTO (PPO)
================================================================================
1. Cruce completo de Cotizaciones, Fundamentales y NLP.
2. Mapeo numérico del sentimiento (-1, 0, 1).
3. Limpieza y rellenado de NAs para optimización del modelo.
================================================================================
"""


"""
================================================================================
GENERADOR DE DATASETS: CRUDO Y ENTRENAMIENTO + CHECK DATA
================================================================================
"""

import pandas as pd
import numpy as np
import os

print("="*70)
print("INICIANDO PROCESAMIENTO DE DATASETS PPO")
print("="*70)

# ==============================================================================
# RUTAS DE LOS ARCHIVOS
# ==============================================================================
ARCHIVO_COTIZACIONES = "cotiz_S&P_technical_complete.csv"
ARCHIVO_FUNDAMENTALES = "Analisis_Fundamental/S&P_v2/fundamentales_limpios.csv"
ARCHIVO_NLP = "resultados_NLP/bardsai_finance/bardsai_finance_results.csv"

DIR_SALIDA = "GA_Env"
os.makedirs(DIR_SALIDA, exist_ok=True)
RUTA_CRUDO = f"{DIR_SALIDA}/dataset__crudo_ga_IBEX_v2.csv"
RUTA_ENTRENAMIENTO = f"{DIR_SALIDA}/dataset_entrenamiento_ga_IBEX_v2.csv" #MUY IMPORTANTE

# ==============================================================================
# PASO 1: PROCESAMIENTO PREVIO DE NLP (SENTIMIENTO)
# ==============================================================================
print("1. Procesando y mapeando sentimientos NLP...")
if os.path.exists(ARCHIVO_NLP):
    df_nlp = pd.read_csv(ARCHIVO_NLP)
    
    if 'Fecha_Publicacion' in df_nlp.columns:
        df_nlp.rename(columns={'Fecha_Publicacion': 'Date'}, inplace=True)
    
   # Limpiamos Tickers por si traen espacios ocultos y le quitamos el ".MC"
    df_nlp['Ticker'] = df_nlp['Ticker'].astype(str).str.replace('.MC', '', regex=False).str.strip()
    # Ajuste manual para excepciones conocidas entre tus fuentes
    df_nlp['Ticker'] = df_nlp['Ticker'].replace({'ADXR': 'ADX'})
    
    # Parseo de fechas forzando UTC y luego eliminándolo
    df_nlp['Date'] = pd.to_datetime(df_nlp['Date'], format='mixed', errors='coerce', utc=True).dt.tz_localize(None).dt.normalize()
    df_nlp = df_nlp.dropna(subset=['Date'])

    # Mapeo de Sentimiento
    mapeo_sent = {'Negativo': -1, 'Neutral': 0, 'Positivo': 1}
    df_nlp['Sentimiento_Num'] = df_nlp['Sentimiento'].map(mapeo_sent)
    
    df_nlp_diario = df_nlp.groupby(['Ticker', 'Date'])['Sentimiento_Num'].mean().reset_index()
    
    # --- CHECK DATA NLP ---
    print("\n[CHECK DATA] -> ESTADO DEL NLP ANTES DEL CRUCE:")
    print(f"  - Registros válidos NLP: {len(df_nlp_diario)}")
    print(f"  - Valores de sentimiento encontrados: {df_nlp['Sentimiento'].unique()}")
    print(f"  - Tickers únicos en NLP (Muestra): {df_nlp_diario['Ticker'].unique()[:5]}")
    print(f"  - Rango fechas NLP: {df_nlp_diario['Date'].min().date()} a {df_nlp_diario['Date'].max().date()}\n")

else:
    print("⚠️ Archivo NLP no encontrado.")
    df_nlp_diario = pd.DataFrame(columns=['Ticker', 'Date', 'Sentimiento_Num'])

# ==============================================================================
# PASO 2: CARGA DE COTIZACIONES Y CÁLCULOS DINÁMICOS
# ==============================================================================
print("2. Cargando Cotizaciones y calculando Rentabilidad/Volatilidad...")
df_base = pd.read_csv(ARCHIVO_COTIZACIONES)
df_base['Ticker'] = df_base['Ticker'].astype(str).str.strip()
df_base['Date'] = pd.to_datetime(df_base['Date'], utc=True).dt.tz_localize(None).dt.normalize()
df_base = df_base.sort_values(['Ticker', 'Date'])

rentabilidad = df_base.groupby('Ticker')['Close'].pct_change()
volatilidad = df_base.groupby('Ticker')['Close'].transform(lambda x: x.pct_change().rolling(window=20).std())

if 'Volume' in df_base.columns:
    idx_volume = df_base.columns.get_loc('Volume')
    df_base.insert(idx_volume + 1, 'Rentabilidad_Diaria', rentabilidad)
    df_base.insert(idx_volume + 2, 'Volatilidad_Diaria', volatilidad)

# --- CHECK DATA COTIZACIONES ---
print("[CHECK DATA] -> ESTADO DE LAS COTIZACIONES:")
print(f"  - Tickers únicos en Cotizaciones (Muestra): {df_base['Ticker'].unique()[:5]}")
print(f"  - Rango fechas Cotiz: {df_base['Date'].min().date()} a {df_base['Date'].max().date()}\n")

# ==============================================================================
# PASO 3: PROCESAMIENTO DE FUNDAMENTALES (AJUSTE TEMPORAL Y ANCLAJE)
# ==============================================================================
print("3. Alineando fundamentales sobre calendario diario de cotizaciones...")
df_fund = pd.read_csv(ARCHIVO_FUNDAMENTALES)
df_fund['Ticker'] = df_fund['Ticker'].astype(str).str.strip()
df_fund['Date'] = pd.to_datetime(df_fund['Date'], errors='coerce').dt.normalize()
df_fund = df_fund.dropna(subset=['Date']).sort_values(['Ticker', 'Date']).reset_index(drop=True)

df_base['Date'] = pd.to_datetime(df_base['Date'], errors='coerce').dt.normalize()
df_base = df_base.dropna(subset=['Date']).sort_values(['Ticker', 'Date']).reset_index(drop=True)

lista_fund = []

tickers_universo = df_base['Ticker'].dropna().unique()

for ticker in tickers_universo:
    base_ticker = (
        df_base.loc[df_base['Ticker'] == ticker, ['Ticker', 'Date']]
        .sort_values('Date')
        .reset_index(drop=True)
    )

    fund_ticker = (
        df_fund.loc[df_fund['Ticker'] == ticker]
        .sort_values('Date')
        .reset_index(drop=True)
    )

    if fund_ticker.empty:
        continue

    alineado = pd.merge_asof(
        base_ticker,
        fund_ticker,
        on='Date',
        direction='backward'
    )

    alineado['Ticker'] = ticker
    lista_fund.append(alineado)

df_fund_final = pd.concat(lista_fund, ignore_index=True)
# ==============================================================================
# PASO 4: CREACIÓN DEL DATASET CRUDO Y ALINEACIÓN
# ==============================================================================
print("4. Generando Dataset CRUDO...")

# 1. Merge
df_crudo = pd.merge(df_base, df_nlp_diario, on=['Ticker', 'Date'], how='left')
df_crudo = pd.merge(df_crudo, df_fund_final, on=['Ticker', 'Date'], how='left')

# 2. DEFINICIÓN SEGURA DE COLUMNAS A RELLENAR (Solo financieras)
# Quitamos descriptivas y columnas identificativas que no deben propagarse
# ni quedarse en el dataset final.
descriptivas_a_excluir = [
    'Ticker', 'Date', 'Instrument', 'instrument', 'Nombre', 'Indice',
    'TRBC_Sector', 'Grupo_Contable', 'Sector', 'Industria', 'Nombre_fund', 'Indice_fund'
]

# Creamos la lista real de columnas financieras que existen en df_crudo
cols_fundamentales = [c for c in df_fund_final.columns if c not in descriptivas_a_excluir and c in df_crudo.columns]

# 3. Aplicar ffill SOLO sobre las columnas fundamentales numéricas
print(f"   -> Aplicando ffill sobre {len(cols_fundamentales)} columnas financieras...")
df_crudo[cols_fundamentales] = df_crudo.groupby('Ticker')[cols_fundamentales].ffill()

# 4. Limpiar cualquier resto de merge
df_crudo = df_crudo.loc[:, ~df_crudo.columns.str.endswith(('_fund', '_y'))]

# 5. Eliminar columna instrument / Instrument si existe
cols_instrument = [c for c in ['instrument', 'Instrument'] if c in df_crudo.columns]
if cols_instrument:
    df_crudo.drop(columns=cols_instrument, inplace=True)

# 6. Filtrado final
df_crudo = df_crudo[df_crudo['Date'] >= '2021-01-01'].copy()
# ==============================================================================
# PASO 5: CREACIÓN DEL DATASET DE ENTRENAMIENTO (PROCESADO)
# ==============================================================================
print("5. Aplicando ajustes lógicos para el Dataset de Entrenamiento...")
df_train = df_crudo.copy()

# A) Borrado exacto de variables
vars_a_borrar = [
    'SCORE_FINAL', 'Anio', 'Trimestre', 'RIC_fund', 'Nombre_fund', 
    'Indice_fund', 'TRBC_Sector', 'Grupo_Contable', 'Net Interest Income', 
    'Fecha', 'Depreciation And Amortization', 'Cash', 
    'Capital Expenditures, Cumulative', 'Sector_fund', 'Industria_fund', 'TRBC_Industry', 'Instrument', 'instrument'
]
df_train.drop(columns=[c for c in vars_a_borrar if c in df_train.columns], inplace=True)

# B) Forward Fill de Fundamentales PRIMERO
df_train = df_train.sort_values(['Ticker', 'Date'])
cols_fundamentales = [c for c in df_fund_final.columns if c in df_train.columns and c not in ['Ticker', 'Date', 'Instrument', 'instrument']]

if cols_fundamentales:
    df_train[cols_fundamentales] = df_train.groupby('Ticker')[cols_fundamentales].ffill()

# C) ⚠️ CORRECCIÓN CLAVE: Tratamiento del Sentimiento NLP (Arrastre y Decaimiento)
# Esto DEBE hacerse antes de rellenar los huecos restantes con ceros.
if 'Sentimiento_Num' in df_train.columns:
    # 1. Arrastramos la última noticia conocida hacia adelante
    df_train['Sentimiento_Num'] = df_train.groupby('Ticker')['Sentimiento_Num'].ffill()
    
    # 2. Aplicamos un decaimiento exponencial para que el impacto se diluya (span=5 días)
    # Rellenamos los NaNs iniciales (antes de la 1ª noticia) con 0 para que la fórmula matemática no falle.
    df_train['Sentimiento_Num'] = df_train['Sentimiento_Num'].fillna(0)
    df_train['Sentimiento_Num'] = df_train.groupby('Ticker')['Sentimiento_Num'].transform(lambda x: x.ewm(span=5, adjust=False).mean())

# D) Ahora sí, poner valor Cero a indicadores técnicos con NA (primeras ventanas)
cols_no_tecnicas = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Rentabilidad_Diaria', 'Volatilidad_Diaria', 'Sentimiento_Num'] + cols_fundamentales
cols_tecnicas = [c for c in df_train.columns if c not in cols_no_tecnicas]

df_train[cols_tecnicas] = df_train[cols_tecnicas].fillna(0)

# E) Limpiar las primeras filas sin Volatilidad
df_train = df_train.dropna(subset=['Volatilidad_Diaria'])

# F) Relleno de seguridad final
df_train.fillna(0, inplace=True)

# Exportar
df_train.to_csv(RUTA_ENTRENAMIENTO, index=False)
print("="*70)
print(f"✅ PROCESO FINALIZADO CON ÉXITO")
print(f"📁 Dataset CRUDO: {RUTA_CRUDO}")
print(f"📁 Dataset ENTRENAMIENTO: {RUTA_ENTRENAMIENTO}")
print("="*70)
