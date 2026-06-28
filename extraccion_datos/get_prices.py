#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# get_prices.py
from datetime import datetime
import lseg.data as ld
import pandas as pd
import time

"""
https://github.com/LSEG-API-Samples/Example.DataLibrary.Python/blob/lseg-data-examples/Examples/2-Content/2.01-HistoricalPricing/EX-2.01.02-HistoricalPricing-ParallelRequests.ipynb
https://github.com/LSEG-API-Samples/Example.DataLibrary.Python/blob/lseg-data-examples/Examples/1-Access/EX-1.01.02-GetHistory.ipynb
"""

# ==============================================================================
# CRONOMETRO Y PARAMETROS
# ==============================================================================

#Cronometro
tiempo_inicio = time.time()

#PARAMETROS
start_date = "2020-01-01"
end_date = "2025-12-31"
frequency = "1D"
df_SP = pd.read_csv("df_u_lseg.csv")

# ==============================================================================
# 1. DEFINICIÓN DE LA FUNCIÓN DE OBTENCIÓN DE DATOS
# ==============================================================================

"""
Descarga histórico de cotizaciones para el universo completo.

Parameters
----------
df_SP : DataFrame
    Debe contener una columna 'ric'.
    start_date : str (YYYY-MM-DD)
    end_date : str or None
        Si None, usa fecha de hoy.
    frequency : str
        Intervalo temporal (por ej., 'D' diario, 'W' semanal, etc. según LSEG).

    Returns
    -------
    df_precios : DataFrame
        Incluye columnas 'ric', 'Date' y campos de precio (según API).
    meta_precios : dict
        Metadatos de la descarga.
"""
def get_price_history(df_SP,
                      start_date=start_date, 
                      end_date=end_date,
                      frequency=frequency):
    """
    Descarga histórico de cotizaciones para el universo completo y añade 
    TODAS las columnas del dataframe original, asegurando 'RIC' en mayúsculas.
    """
    # 1. ESTANDARIZAR EL UNIVERSO
    # Convertimos 'ric' a mayúsculas ('RIC') en la tabla original para que todo cuadre
    if "ric" in df_SP.columns:
        df_SP = df_SP.rename(columns={"ric": "RIC"})
        
    lista_rics = df_SP["RIC"].tolist()

    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"Descargando histórico de precios para {len(lista_rics)} RICs...")
    print(f"Rango: {start_date} -> {end_date}, frecuencia: {frequency}")

    # Descarga de datos
    df_hist = ld.get_history(
        universe=lista_rics,
        start=start_date,
        end=end_date,
        interval=frequency,
    )

    # 2. APLANAR LA TABLA DE PRECIOS
    df_precios = df_hist.stack(level=1, future_stack=True).reset_index()

    # Identificamos los nombres actuales para curarnos en salud contra el KeyError
    columnas_actuales = df_precios.columns.tolist()
    
    if "DATE" in columnas_actuales:
        df_precios = df_precios.rename(columns={"DATE": "Date"})
        
    # LSEG puede llamar a la columna de RICs de varias formas tras el stack. 
    # La buscamos y la forzamos a ser "RIC"
    if "Instrument" in columnas_actuales:
        df_precios = df_precios.rename(columns={"Instrument": "RIC"})
    elif "level_1" in columnas_actuales:
        df_precios = df_precios.rename(columns={"level_1": "RIC"})
    elif "ric" in columnas_actuales:
        df_precios = df_precios.rename(columns={"ric": "RIC"})
    
    # Si por algún motivo extraño la columna sigue sin llamarse 'RIC', 
    # asumimos por estructura matemática que es la segunda columna (índice 1)
    if "RIC" not in df_precios.columns:
        df_precios = df_precios.rename(columns={df_precios.columns[1]: "RIC"})

    # 3. CRUZAR CON TODAS LAS COLUMNAS DEL UNIVERSO
    # Al pasarle todo 'df_SP', se traerá el Nombre, Macro, Micro, Ticker, etc.
    df_precios = pd.merge(
        df_precios, 
        df_SP, 
        on="RIC", 
        how="left"
    )

    meta_precios = {
        "n_filas": int(len(df_precios)),
        "n_columnas": int(df_precios.shape[1]),
        "start_date_solicitada": start_date,
        "end_date_solicitada": end_date,
        "frequency": frequency,
        "rics": lista_rics,
    }

    return df_precios, meta_precios

# ==============================================================================
# 2. CARGA DE DATOS Y MODELO
# ==============================================================================
if __name__ == "__main__":
   
    from universo import build_universe
    from lseg_config import open_session, close_session

    df_u_lseg, _ = build_universe()
    open_session()
    df_p, meta_p = get_price_history(df_SP) #CAMBIAR CADA VEZ QUE SE CAMBIE EL NOMBRE
    print(df_p.head())
    print(meta_p)
"""
# ==============================================================================
# NUEVO: DESCARGA EXCLUSIVA DE ÍNDICES (BENCHMARK REAL)
# ==============================================================================
print("\n--- Iniciando descarga de cotizaciones oficiales de índices ---")

# 1. Definimos los RICs de los índices
rics_benchmarks = [".IBEX", ".IBEXC", ".IBEXS"]

# 2. Descargamos los datos (OHLCV)
df_indices = ld.get_history(
    universe=rics_benchmarks,
    start=start_date,  # Usamos la misma fecha que definiste arriba (2021-01-01)
    end=end_date,      # Usamos la misma fecha (2026-01-01)
    interval=frequency, # 1D
    fields=["TR.OPENPRICE", "TR.HIGHPRICE", "TR.LOWPRICE", "TR.CLOSEPRICE", "TR.ACCUMULATEDVOLUME"]
)

if df_indices is not None and not df_indices.empty:
    # 3. Guardamos en el archivo específico solicitado
    archivo_indices = "cotiz_indices.csv"
    df_indices.to_csv(archivo_indices, index=True, encoding="utf-8-sig")
    
    print(f"¡ÉXITO! Los datos de los índices se han guardado en '{archivo_indices}'.")
    print(f"Columnas detectadas: {list(df_indices.columns)}")
else:
    print("❌ ERROR: No se pudieron descargar los datos de los índices oficiales.")
    close_session()
    """
# ==============================================================================
# 3. PARAR CRONOMETRO
# ==============================================================================    
 # Guardamos la hora de finalización
tiempo_fin = time.time()

# Calculamos la diferencia
tiempo_total_segundos = tiempo_fin - tiempo_inicio

# Traducimos los segundos a formato horas, minutos y segundos
horas = int(tiempo_total_segundos // 3600)
minutos = int((tiempo_total_segundos % 3600) // 60)
segundos = int(tiempo_total_segundos % 60)

# Mostramos el mensaje final
print(f"⏱️ El script ha tardado: {horas} horas, {minutos} minutos y {segundos} segundos.")

"""
⏱️ El script ha tardado: 0 horas, 0 minutos y 38 segundos
"""
df_p.to_csv('df_p_IBEX_2020.csv', index=False)
