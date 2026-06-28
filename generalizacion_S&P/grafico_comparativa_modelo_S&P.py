# -*- coding: utf-8 -*-
"""
Script Definitivo S&P 500: Todo desde portfolio_evolution_base100.csv
Marco gris claro, base 100 estricta y formato listo para LaTeX.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# 1. PARÁMETROS DE CONFIGURACIÓN
# ==============================================================================
MERCADO = "S&P"  
# [!] CAMBIA ESTO por los nombres de tus carpetas del experimento S&P 500
EXPERIMENTO_1 = "GA_MultiStock_v25.2"
EXPERIMENTO_2 = "GA_MultiStock_v28"

BASE_DIR = Path("./GA_Env") 
DIR_EXP_1 = BASE_DIR / "experimentos/GA_MultiStock_v25.2/backtest_S&P_v2"
DIR_EXP_2 = BASE_DIR / "experimentos/GA_MultiStock_v28/backtest_S&P_v2"
DIR_SALIDA = BASE_DIR / "Comparativas"
NOMBRE_CSV_MODELOS = "portfolio_evolution_base100.csv"
COL_FECHA = "Date"

# Nombres EXACTOS en tu CSV del S&P
COL_AGENTES = "Agente AG"
COL_BH = "Buy & Hold"
COL_SP = "S&P 500" # <--- 

TITULO_GRAFICO = "Evolución Normalizada (Base 100) — Comparativa Modelos S&P 500"
NOMBRE_ARCHIVO_SALIDA = "comparativa_modelos_vs_sp500.png"

# ==============================================================================
# 3. CARGA Y PROCESAMIENTO (TODO DESDE EL MISMO CSV)
# ==============================================================================
print(f"Cargando datos para el mercado: {MERCADO}...")
DIR_SALIDA.mkdir(parents=True, exist_ok=True)

# Cargar Experimento 1 (Trae el modelo base, B&H y el índice S&P)
df1 = pd.read_csv(DIR_EXP_1 / NOMBRE_CSV_MODELOS, parse_dates=[COL_FECHA]).set_index(COL_FECHA).sort_index()
df1 = df1.loc[~df1.index.duplicated(keep='first')]

s_base = df1[COL_AGENTES].rename('Modelo_Base')
s_bh = df1[COL_BH].rename('Base100_BH')
s_sp = df1[COL_SP].rename('Base100_SP')

# Cargar Experimento 2 (Solo el Agente Dinámico)
df2 = pd.read_csv(DIR_EXP_2 / NOMBRE_CSV_MODELOS, parse_dates=[COL_FECHA]).set_index(COL_FECHA).sort_index()
df2 = df2.loc[~df2.index.duplicated(keep='first')]
s_dinamico = df2[COL_AGENTES].rename('Modelo_Dinamico')

# Cruce limpio instantáneo
df_cruzado = pd.concat([s_base, s_dinamico, s_bh, s_sp], axis=1, join='inner')

# Asegurar Base 100 estricta en la primera fila de la comparativa
for col in df_cruzado.columns:
    df_cruzado[col] = (df_cruzado[col] / df_cruzado[col].iloc[0]) * 100.0

# ==============================================================================
# 5. GENERACIÓN DEL GRÁFICO 
# ==============================================================================
plt.style.use('seaborn-v0_8-white')
fig, ax = plt.subplots(figsize=(16, 7), facecolor='white')

# Líneas de los Modelos (Gruesas)
ax.plot(df_cruzado.index, df_cruzado['Modelo_Base'], label="Algoritmo Genético (Base)", color="#4c72b0", linewidth=2.5)
ax.plot(df_cruzado.index, df_cruzado['Modelo_Dinamico'], label="Algoritmo Genético (Dinámico)", color="#c44e52", linewidth=2.5)

# Líneas de Benchmarks (Finas / Discontinuas)
ax.plot(df_cruzado.index, df_cruzado['Base100_BH'], label="Buy & Hold (Equiponderado)", color="#dd8452", linewidth=1.5, linestyle="--")
ax.plot(df_cruzado.index, df_cruzado['Base100_SP'], label="Índice S&P 500", color="#55a868", linewidth=1.5, linestyle="--")

# Línea base en 100 y límites del eje Y
ax.axhline(100, color="black", linestyle=":", alpha=0.6)
ax.set_ylim(60, 160)

# Ocultar la etiqueta del valor mínimo (80) y máximo (180)
ax.set_yticks(ax.get_yticks())
labels = ax.get_yticklabels()
labels[0].set_visible(False)
labels[-1].set_visible(False)

# Cambiar el color del marco a gris claro
for spine in ax.spines.values():
    spine.set_color('lightgray')

# Estética final
ax.set_title(TITULO_GRAFICO, fontsize=14, fontweight="bold", pad=15)
ax.legend(fontsize=10, ncol=2, loc="upper left", frameon=True, edgecolor="lightgray")
ax.grid(True, alpha=0.3)
fig.tight_layout()

ruta_salida = DIR_SALIDA / NOMBRE_ARCHIVO_SALIDA
plt.savefig(ruta_salida, dpi=300, bbox_inches="tight", facecolor='white')
plt.close()

print(f"¡Listo! Gráfico S&P 500 guardado en: {ruta_salida}")