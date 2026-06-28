#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
analisis_tecnico.py
-------------------
Calcula todos los indicadores técnicos sobre df_p_limpio (OHLCV ya limpio).
Asume que df_p_limpio tiene las columnas:
    ric, nombre, sector, Fecha, Precio_Apertura, Precio_Alto,
    Precio_Bajo, Precio_Cierre, Volumen
    
REFERENCIA 
https://github.com/LSEG-API-Samples/Workflow.LD.Python.TechnicalAnalysis/blob/main/TA_Workflow.ipynb
"""

# ==============================================================================
# 1. DEFINICIÓN DE FUNCIÓN DE CALCULO INDICADORES TÉCNICOS
# ==============================================================================

import pandas as pd
# Importaciones de la librería 'ta' para todos los indicadores
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import time

# ==============================================================================
# CRONÓMETRO
# ==============================================================================
tiempo_inicio = time.time()


print("1. Cargando y preparando datos...")
# Cargar los datos originales
df = pd.read_csv("cotiz_IBEX_2020_limpio.csv")

# Asegurar que la fecha es un objeto datetime y ordenar cronológicamente por empresa
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Ticker', 'Date'])

def add_ta_indicators(group):
    """
    Función que calcula todos los indicadores técnicos para una sola empresa.
    """
    # Extraemos las series de datos que necesitan las fórmulas
    close = group['Close']
    high = group['High']
    low = group['Low']
    volume = group['Volume']
    
    # --- 1. INDICADORES DE MOMENTO ---
    # RSI (14)
    group['RSI_14'] = RSIIndicator(close=close, window=14).rsi()
    
    # Oscilador Estocástico (14, 3)
    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    group['Stoch_k'] = stoch.stoch()
    group['Stoch_d'] = stoch.stoch_signal()

    # --- 2. INDICADORES DE TENDENCIA ---
    # MACD (12, 26, 9)
    macd = MACD(close=close)
    group['MACD'] = macd.macd()
    group['MACD_Signal'] = macd.macd_signal()
    group['MACD_Hist'] = macd.macd_diff()
    
    # ADX (14) - Fuerza de la tendencia
    adx = ADXIndicator(high=high, low=low, close=close, window=14)
    group['ADX_14'] = adx.adx()
    
    # Medias Móviles Simples (50 y 200 periodos)
    group['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
    group['SMA_50'] = SMAIndicator(close=close, window=50).sma_indicator()
    group['SMA_200'] = SMAIndicator(close=close, window=200).sma_indicator()

    # --- 3. INDICADORES DE VOLATILIDAD ---
    # Bandas de Bollinger (20, 2)
    bb = BollingerBands(close=close, window=20, window_dev=2)
    group['SMA_20'] = bb.bollinger_mavg()
    group['BB_Upper'] = bb.bollinger_hband()
    group['BB_Lower'] = bb.bollinger_lband()
    
    # ATR (14) - Average True Range
    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    group['ATR_14'] = atr.average_true_range()

    # --- 4. INDICADORES DE VOLUMEN ---
    # OBV - On-Balance Volume
    obv = OnBalanceVolumeIndicator(close=close, volume=volume)
    group['OBV'] = obv.on_balance_volume()
    
    return group

print("2. Calculando indicadores para las empresas (esto puede tardar unos segundos)...")
# Aplicamos la función agrupando por 'Ticker'
df_indicators = df.groupby('Ticker', group_keys=False).apply(add_ta_indicators)


# --- PARCHE DE SEGURIDAD SPYDER ---
# Evitamos que pandas elimine la columna 'Ticker' al hacer el groupby
if 'Ticker' not in df_indicators.columns:
    if 'Ticker' in df_indicators.index.names:
        df_indicators = df_indicators.reset_index()
    else:
        df_indicators['Ticker'] = df['Ticker'].values
# ---------------------------------------

# --- REORDENAR COLUMNAS ---
# Creamos una lista con 'Date' y 'Ticker' al principio
nuevas_columnas = ['Date', 'Ticker'] 
# Le añadimos el resto de columnas que no sean ni Date ni Ticker
nuevas_columnas += [col for col in df_indicators.columns
                    if col 
                    not in ['Date', 'Ticker']]

# Aplicamos el nuevo orden al DataFrame
df_indicators = df_indicators[nuevas_columnas]
# --------------------------
print("3. Guardando el archivo final...")
# Guardamos el resultado en un nuevo CSV
nombre_archivo = "cotiz_IBEX_2020_completo.csv"
df_indicators.to_csv(nombre_archivo, index=False)


# ==============================================================================
# PARAR CRONÓMETRO
# ==============================================================================
tiempo_fin = time.time()
tiempo_total_segundos = tiempo_fin - tiempo_inicio

horas = int(tiempo_total_segundos // 3600)
minutos = int((tiempo_total_segundos % 3600) // 60)
segundos = int(tiempo_total_segundos % 60)

print(f"Archivo guardado : {nombre_archivo}")