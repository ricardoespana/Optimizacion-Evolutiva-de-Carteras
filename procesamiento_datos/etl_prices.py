#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import time

# ==============================================================================
# CRONÓMETRO
# ==============================================================================
tiempo_inicio = time.time()

# ==============================================================================
# ETL: DE LSEG A OHLCV PLANO ENRIQUECIDO
# ==============================================================================
def etl_prices(precios_path, universo_path, output_path):
    print("1. EXTRACT: Leyendo el archivo de precios...")
    df = pd.read_csv(precios_path, header=0)
    
    print("2. TRANSFORM: Limpiando columnas basura exportadas por LSEG...")
    columnas_basura = ['Ticker', 'nombre', 'sector', 'indice']
    df.drop(columns=[col for col in columnas_basura if col in df.columns], inplace=True)
    
    # ¡LA CLAVE DEL ERROR ESTABA AQUÍ!
    # LSEG llamó 'RIC' a la columna que contiene los nombres de las métricas (OPEN_PRC, etc.).
    # Vamos a renombrarla a 'Metric' para no confundirla con nuestros verdaderos identificadores de empresa.
    if 'RIC' in df.columns:
        df.rename(columns={'RIC': 'Metric'}, inplace=True)
    
    print("3. TRANSFORM: Pasando de formato ancho a largo (Melt)...")
    # Al derretir la tabla, las cabeceras (A3M.MC, SAN.MC) pasan a ser filas. 
    # Las llamamos por su verdadero nombre: 'RIC'.
    df_melted = df.melt(id_vars=['Date', 'Metric'], var_name='RIC', value_name='Value')
    
    print("4. TRANSFORM: Construyendo estructura OHLCV (Pivot)...")
    df_pivot = df_melted.pivot_table(index=['Date', 'RIC'], columns='Metric', values='Value').reset_index()
    
    rename_map = {
        'OPEN_PRC': 'Open',
        'HIGH_1': 'High',
        'LOW_1': 'Low',
        'TRDPRC_1': 'Close',
        'ACVOL_UNS': 'Volume'
    }
    
    cols_existentes = ['Date', 'RIC'] + [col for col in rename_map.keys() if col in df_pivot.columns]
    df_precios = df_pivot[cols_existentes].rename(columns=rename_map)
    
    print("5. TRANSFORM: Eliminando días sin cotización...")
    cols_financieras = [col for col in ['Open', 'High', 'Low', 'Close'] if col in df_precios.columns]
    if cols_financieras:
        df_precios.dropna(subset=cols_financieras, how='all', inplace=True)
    
    print("6. ENRIQUECIMIENTO: Cruzando con el universo (RIC, Nombre, Sector, Industria...)...")
    df_u = pd.read_csv(universo_path)
    
    # Blindaje extra: Eliminamos posibles espacios en blanco (' A3M.MC ') que rompen los cruces
    df_precios['RIC'] = df_precios['RIC'].astype(str).str.strip()
    
    # Aseguramos que la columna del universo también se llame 'RIC' (en mayúsculas) y sin espacios
    if 'ric' in df_u.columns:
        df_u.rename(columns={'ric': 'RIC'}, inplace=True)
    if 'RIC' in df_u.columns:
        df_u['RIC'] = df_u['RIC'].astype(str).str.strip()
    
    # EL JOIN DEFINITIVO: Ahora cruzamos manzanas con manzanas (RIC con RIC)
    df_final = pd.merge(df_precios, df_u, on='RIC', how='left')
    
    print("7. LOAD: Ordenando columnas para la tabla plana definitiva...")
    cols_base = ['Date', 'RIC']
    # Cogemos todas las columnas descriptivas que vengan del universo, excluyendo RIC que ya la hemos puesto
    cols_descriptivas = [col for col in df_u.columns if col != 'RIC']
    cols_ohlcv = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in df_final.columns]
    
    df_final = df_final[cols_base + cols_descriptivas + cols_ohlcv]
    
    df_final['Date'] = pd.to_datetime(df_final['Date'])
    df_final.sort_values(by=['RIC', 'Date'], inplace=True)
    
    df_final.to_csv(output_path, index=False)
    print(f"\n¡Éxito total! Archivo sin NaNs guardado como: {output_path}")
    
    return df_final

# ==============================================================================
# EJECUCIÓN
# ==============================================================================
archivo_precios = 'df_p_IBEX_2020.csv'
archivo_universo = 'df_u_lseg.csv'
archivo_salida = 'cotiz_IBEX_2020_limpio.csv'

df_definitivo = etl_prices(archivo_precios, archivo_universo, archivo_salida)

print("\nVista previa de la tabla plana definitiva:")
print(df_definitivo.head())

# ==============================================================================
# PARAR CRONÓMETRO
# ==============================================================================
tiempo_fin = time.time()
tiempo_total_segundos = tiempo_fin - tiempo_inicio

horas = int(tiempo_total_segundos // 3600)
minutos = int((tiempo_total_segundos % 3600) // 60)
segundos = int(tiempo_total_segundos % 60)

print(f"\n⏱️ El script ha tardado: {horas} horas, {minutos} minutos y {segundos} segundos.")