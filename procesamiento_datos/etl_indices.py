#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETL DE ÍNDICES OFICIALES (BENCHMARK) - VERSIÓN OPTIMIZADA
Extrae únicamente los precios de cierre diarios de cada índice desde un CSV de LSEG.
"""

import pandas as pd
import os

print("="*60)
print("INICIANDO ETL DE ÍNDICES (SOLO CIERRES)")
print("="*60)

ARCHIVO_ENTRADA = "cotiz_indices.csv"
ARCHIVO_SALIDA = "indices_cierre.csv"

if not os.path.exists(ARCHIVO_ENTRADA):
    print(f"❌ Error: No se encontró el archivo '{ARCHIVO_ENTRADA}'.")
    exit()

print("1. Leyendo y analizando estructura (MultiIndex)...")
# Leemos el archivo indicando explícitamente las dos filas de cabecera y la columna de índice
df = pd.read_csv(ARCHIVO_ENTRADA, header=[0, 1], index_col=0)

print("2. Extrayendo los precios de cierre...")
# Usamos .xs para extraer de un plumazo solo las columnas que se llaman 'Close Price' en el nivel 1
df_close = df.xs('Close Price', axis=1, level=1)

print("3. Renombrando Tickers y transformando la tabla...")
# Mapeo de RICs a nombres amigables
MAPEO_INDICES = {
    ".IBEX": "IBEX 35", 
    ".IBEXC": "IBEX_MEDIUM_CAP", 
    ".IBEXS": "IBEX_SMALL_CAP"
}
df_close.columns = [MAPEO_INDICES.get(col, col) for col in df_close.columns]

# Transformamos la tabla a formato largo (Date, Ticker, Close)
df_long = df_close.reset_index().melt(
    id_vars='Date', 
    var_name='Ticker', 
    value_name='Close'
)

print("4. Limpiando datos nulos y ordenando...")
# Eliminamos los días sin cotización (fines de semana, festivos) y ordenamos
df_long.dropna(subset=['Close'], inplace=True)
df_long['Date'] = pd.to_datetime(df_long['Date'], errors='coerce')
df_long = df_long.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# Exportamos el resultado final
df_long.to_csv(ARCHIVO_SALIDA, index=False)

print("\n" + "="*60)
print(f"✅ ¡ÉXITO! Datos de cierre guardados en '{ARCHIVO_SALIDA}'.")
print("="*60)
print("Resumen de datos procesados (Días cotizados por índice):")
print(df_long.groupby('Ticker')['Date'].count().to_string())