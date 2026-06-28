#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
# ==============================================================================
# 1.COMPARATIVAS ANUALES DE LAS METRICAS POR ÍNDICE Y SECTOR
# ==============================================================================

# 1. Cargar los datasets
df_u = pd.read_csv('df_u.csv')
df_noticias = pd.read_csv('dataset_2021_2026_eleconomista.csv') #CAMBIAR SEGUN DATASET

# 2. Recuento de cuántas noticias hay por cada Ticker en el dataset extraído
recuento_noticias = df_noticias.groupby('Ticker').size().reset_index(name='Total_Noticias')

# 3. Hacer un "Left Merge" (Cruce por la izquierda)
# Esto mantiene TODAS las filas de tu universo (df_u) y le pega la columna de recuento.
df_check = pd.merge(df_u, recuento_noticias, on='Ticker', how='left')

# 4. Rellenar vacíos
# Si una empresa de df_u no existiera en df_noticias, su recuento sería NaN. Lo pasamos a 0.
df_check['Total_Noticias'] = df_check['Total_Noticias'].fillna(0).astype(int)

# 5. Filtrar las empresas que se quedaron a cero
empresas_sin_datos = df_check[df_check['Total_Noticias'] == 0]

# ==============================================================================
#2. RESULTADOS
# ==============================================================================

print("--- REPORTE DE EXTRACCIÓN ---")
print(f"Total de acciones en el universo: {len(df_u)}")
print(f"Total de acciones con noticias: {len(df_check[df_check['Total_Noticias'] > 0])}")
print(f"Total de acciones SIN noticias: {len(empresas_sin_datos)}")

if len(empresas_sin_datos) > 0:
    print("\n[!] ALERTA: Faltan datos para las siguientes empresas:")
    # Imprime los Tickers y nombres de las que han fallado
    print(empresas_sin_datos[['Ticker', 'nombre']])
else:
    print("\n[ÉXITO] ¡Todas las empresas del universo tienen información asociada!")