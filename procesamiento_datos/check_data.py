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
# 1. CHECK DE LOS DATOS (ADAPTADO A LSEG)
# ==============================================================================
print("1. Cargando el universo de empresas (df_u_lseg.csv)...")
# Usamos el archivo del universo correcto
df_u = pd.read_csv('df_S&P.csv')

# Aseguramos limpieza y extraemos el set de RICs a auditar
df_u['RIC'] = df_u['RIC'].astype(str).str.strip()
rics_universo = set(df_u['RIC'])

print("2. Simulando el proceso ETL sobre df_p.csv...")
# Leemos en formato plano (header=0)
df = pd.read_csv('df_p_S&P.csv', header=0)

# Limpiamos columnas basura
columnas_basura = ['Ticker', 'nombre', 'sector', 'indice']
df.drop(columns=[col for col in columnas_basura if col in df.columns], inplace=True)

# Renombramos la columna RIC a Metric para no confundirla con las cabeceras de las empresas
if 'RIC' in df.columns:
    df.rename(columns={'RIC': 'Metric'}, inplace=True)

# Transformamos de formato ancho a largo (Melt)
df_melted = df.melt(id_vars=['Date', 'Metric'], var_name='RIC', value_name='Value')
df_melted['RIC'] = df_melted['RIC'].astype(str).str.strip()

# Calculamos cuántas empresas distintas han venido en el archivo bruto
rics_antes_drop = set(df_melted['RIC'].unique())

print("-" * 50)
print(f"RICs extraídos del archivo bruto: {len(rics_antes_drop)}")

# Identificamos si falta alguna empresa del universo DESDE EL PRINCIPIO
ni_siquiera_estan = rics_universo - rics_antes_drop
if ni_siquiera_estan:
    print(f"❌ {len(ni_siquiera_estan)} empresas ni siquiera vienen en el CSV bruto de LSEG:")
    for r in sorted(ni_siquiera_estan):
        print(f"   - {r}")
else:
    print("✅ Todas las empresas del universo están presentes en el archivo bruto.")

print("\n3. Construyendo tabla OHLCV y aplicando el filtro DropNA...")
# Pivotamos a la estructura final
df_pivot = df_melted.pivot_table(index=['Date', 'RIC'], columns='Metric', values='Value').reset_index()

rename_map = {
    'OPEN_PRC': 'Open', 'HIGH_1': 'High', 'LOW_1': 'Low',
    'TRDPRC_1': 'Close', 'ACVOL_UNS': 'Volume'
}

cols_existentes = ['Date', 'RIC'] + [col for col in rename_map.keys() if col in df_pivot.columns]
df_precios = df_pivot[cols_existentes].rename(columns=rename_map)

# Aplicamos el DropNA (Eliminar días sin cotización)
cols_financieras = [col for col in ['Open', 'High', 'Low', 'Close'] if col in df_precios.columns]
if cols_financieras:
    df_precios_limpio = df_precios.dropna(subset=cols_financieras, how='all')
else:
    df_precios_limpio = df_precios.copy()

rics_despues_drop = set(df_precios_limpio['RIC'].unique())

print("-" * 50)
print(f"RICs supervivientes tras el DropNA: {len(rics_despues_drop)}")

# Calculamos cuáles empresas del universo han sido borradas completamente en este paso
# (Usamos intersection para no contar columnas raras o 'Unnamed' como empresas eliminadas)
eliminadas_por_dropna = rics_antes_drop.intersection(rics_universo) - rics_despues_drop

if eliminadas_por_dropna:
    print(f"\n⚠️ El DropNA ha eliminado completamente {len(eliminadas_por_dropna)} empresas de tu universo porque su histórico estaba 100% vacío:")
    for r in sorted(eliminadas_por_dropna):
        print(f"   - {r}")
else:
    print("\n✅ Ninguna empresa de tu universo fue eliminada por completo durante la limpieza de nulos.")
    
# ==============================================================================
# PARAR CRONÓMETRO
# ==============================================================================
tiempo_fin = time.time()
tiempo_total_segundos = tiempo_fin - tiempo_inicio

horas = int(tiempo_total_segundos // 3600)
minutos = int((tiempo_total_segundos % 3600) // 60)
segundos = int(tiempo_total_segundos % 60)

"""
RICs extraídos del archivo bruto: 85
❌ 2 empresas ni siquiera vienen en el CSV bruto de LSEG:
   - ENAG.MC
   - NTH.MC
"""