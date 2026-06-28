#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Carga del dataset maestro (el que generaste con LSEG y Yahoo)
# Si aún no lo has guardado, usa el DataFrame resultante del cruce
df = pd.read_csv('universo_maestro_sectores.csv')

# 2. LIMPIEZA PARA COMPARACIÓN (Evitar que "Utilities" vs "utilities" cuente como error)
df['sector_orig'] = df['sector'].str.strip().str.upper()
df['sector_lseg'] = df['Sector_LSEG'].str.strip().str.upper()

# 3. CREACIÓN DE LA MATRIZ DE CONTINGENCIA
# Esto muestra cuántas veces el Sector A coincide con el Sector B
contingency_matrix = pd.crosstab(df['sector_orig'], df['sector_lseg'])

# 4. VISUALIZACIÓN: HEATMAP DE DIVERGENCIAS
plt.figure(figsize=(14, 10))
sns.heatmap(contingency_matrix, annot=True, fmt='d', cmap='YlGnBu', cbar=True)
plt.title('Matriz de Correspondencia: Sector Original vs LSEG (TRBC)', fontsize=15)
plt.xlabel('Sectores LSEG', fontsize=12)
plt.ylabel('Sectores Originales (BME-Style)', fontsize=12)
plt.tight_layout()

# Guardar en la carpeta macro para el TFM
ruta_save = 'reporte_eda/macro/05_matriz_divergencias_sectores.png'
os.makedirs(os.path.dirname(ruta_save), exist_ok=True)
plt.savefig(ruta_save, dpi=300)
plt.show()

# 5. EXTRACCIÓN DE "CASOS EN CONFLICTO"
# Filtramos empresas donde las etiquetas son semánticamente muy distintas
print("\n--- Top 10 Casos de Divergencia Significativa ---")
# Ejemplo: Casos donde LSEG dice 'Industrials' pero tu original dice 'Tecnología'
divergentes = df[df['sector_orig'] != df['sector_lseg']][['Ticker', 'sector', 'Sector_LSEG', 'Sector_GICS_Yahoo']]
print(divergentes.head(10))