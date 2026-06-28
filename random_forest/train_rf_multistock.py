# -*- coding: utf-8 -*-
"""
Script Definitivo: Evolución de Modelos IA vs Benchmarks.
- Extrae el Buy & Hold directamente del CSV del experimento.
- Inclusión de Multi-Índice (Medium y Small Cap).
- Eje Y fijo (80-180) y formato para LaTeX.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# 1. PARÁMETROS DE CONFIGURACIÓN
# ==============================================================================
MERCADO = "IBEX"  
EXPERIMENTO_1 = "GA_MultiStock_v25.2"
EXPERIMENTO_2 = "GA_MultiStock_v28"

BASE_DIR = Path("./GA_Env") 
DIR_EXP_1 = BASE_DIR / "experimentos/GA_MultiStock_v25.2/backtest_results"
DIR_EXP_2 = BASE_DIR / "experimentos/GA_MultiStock_v28/backtest_results"
DIR_SALIDA = BASE_DIR / "Comparativas"

NOMBRE_CSV_MODELOS = "portfolio_evolution_base100.csv"
COL_FECHA = "Date"

# Nombres exactos de las columnas en tus archivos CSV
COL_CSV_AGENTES = "GA Agent" 
COL_CSV_BH = "Buy & Hold" # <--- REVISA QUE SE LLAME ASÍ EXACTAMENTE EN TU CSV

RUTA_ARCHIVO_INDICES = BASE_DIR / "data/indices_cierre.csv"

# ==============================================================================
# 2. LÓGICA CONDICIONAL DE MERCADO Y MULTI-ÍNDICE
# ==============================================================================
if MERCADO == "IBEX":
    INDICES_A_MOSTRAR = ["IBEX 35", "IBEX MEDIUM CAP", "IBEX SMALL CAP"]
    TITULO_GRAFICO = "Evolución Normalizada (Base 100) — Comparativa Modelos IBEX"
    NOMBRE_ARCHIVO_SALIDA = "comparativa_modelos_vs_ibex.png"
elif MERCADO == "S&P":
    INDICES_A_MOSTRAR = ["S&P"]
    TITULO_GRAFICO = "Evolución Normalizada (Base 100) — Comparativa Modelos S&P"
    NOMBRE_ARCHIVO_SALIDA = "comparativa_modelos_vs_sp.png"
else:
    raise ValueError("El parámetro MERCADO debe ser 'IBEX' o 'S&P'")

LEYENDA_EXP1 = "Algoritmo Genético (Base)"
LEYENDA_EXP2 = "Algoritmo Genético (Dinámico)"
LEYENDA_BH = "Buy & Hold (Equiponderado)"

# ==============================================================================
# 3. CARGA Y PROCESAMIENTO (EXTRAE EL B&H DIRECTAMENTE)
# ==============================================================================
print(f"Preparando cruce de datos para el mercado: {MERCADO}")
DIR_SALIDA.mkdir(parents=True, exist_ok=True)

# 3.1 Modelos y B&H: Cargar del Exp 1 (Agente y Buy & Hold)
df1 = pd.read_csv(DIR_EXP_1 / NOMBRE_CSV_MODELOS, parse_dates=[COL_FECHA]).set_index(COL_FECHA).sort_index()
df1 = df1.loc[~df1.index.duplicated(keep='first')]

s1 = df1[COL_CSV_AGENTES].rename('Modelo_Base')
s_bh = df1[COL_CSV_BH].rename('Base100_BH')  # Extraemos el B&H ya calculado

# Cargar del Exp 2 (Solo el Agente Dinámico)
df2 = pd.read_csv(DIR_EXP_2 / NOMBRE_CSV_MODELOS, parse_dates=[COL_FECHA]).set_index(COL_FECHA).sort_index()
df2 = df2.loc[~df2.index.duplicated(keep='first')]
s2 = df2[COL_CSV_AGENTES].rename('Modelo_Dinamico')

calendario_maestro = s1.index

# 3.2 Índices: Carga dinámica
df_indices_raw = pd.read_csv(RUTA_ARCHIVO_INDICES, parse_dates=[COL_FECHA])
dict_indices = {}

for target_index in INDICES_A_MOSTRAR:
    df_item = df_indices_raw[df_indices_raw['Ticker'] == target_index].set_index(COL_FECHA).sort_index()
    df_item = df_item.loc[~df_item.index.duplicated(keep='first')]
    
    serie_close = df_item['Close'].reindex(calendario_maestro).ffill().bfill()
    dict_indices[f'Base100_{target_index}'] = (serie_close / serie_close.iloc[0]) * 100.0

# ==============================================================================
# 4. CRUCE MAESTRO
# ==============================================================================
df_cruzado = pd.DataFrame(index=calendario_maestro)
df_cruzado['Modelo_Base'] = s1
df_cruzado['Modelo_Dinamico'] = s2.reindex(calendario_maestro).ffill().bfill()
df_cruzado['Base100_BH'] = s_bh.reindex(calendario_maestro).ffill().bfill()

for key, serie in dict_indices.items():
    df_cruzado[key] = serie

# Verificación de normalización a Base 100 el primer día cruzado
for col in df_cruzado.columns:
    df_cruzado[col] = (df_cruzado[col] / df_cruzado[col].iloc[0]) * 100.0

# ==============================================================================
# 5. GENERACIÓN DEL GRÁFICO (EJE Y 80-180)
# ==============================================================================
plt.style.use('seaborn-v0_8-white')
plt.figure(figsize=(16, 7))

# Líneas Agentes (IA)
plt.plot(df_cruzado.index, df_cruzado['Modelo_Base'], label=LEYENDA_EXP1, color="#4c72b0", linewidth=2.5, linestyle="-")
plt.plot(df_cruzado.index, df_cruzado['Modelo_Dinamico'], label=LEYENDA_EXP2, color="#c44e52", linewidth=2.5, linestyle="-")

# Línea B&H (Extraída de tu CSV)
plt.plot(df_cruzado.index, df_cruzado['Base100_BH'], label=LEYENDA_BH, color="#dd8452", linewidth=1.5, linestyle="--")

# Estilos Índices
estilos_indices = {
    "IBEX 35": {"color": "#55a868", "style": "--"},
    "IBEX MEDIUM CAP": {"color": "black", "style": "--"},
    "IBEX SMALL CAP": {"color": "gray", "style": "--"},
    "S&P": {"color": "#55a868", "style": "--"}
}

for target_index in INDICES_A_MOSTRAR:
    col_name = f'Base100_{target_index}'
    if col_name in df_cruzado.columns:
        config = estilos_indices.get(target_index, {"color": "black", "style": "--"})
        plt.plot(df_cruzado.index, df_cruzado[col_name], label=f"Índice {target_index}", 
                 color=config["color"], linewidth=1.5, linestyle=config["style"])

# Limites requeridos
plt.axhline(100, color="grey", linestyle=":", alpha=0.6)
plt.ylim(80, 180)

# Estética
plt.title(TITULO_GRAFICO, fontsize=14, fontweight="bold", pad=15)
plt.legend(fontsize=10, ncol=2, loc="upper left", frameon=True, edgecolor="lightgray")
plt.grid(True, alpha=0.3)
plt.tight_layout()

ruta_salida = DIR_SALIDA / NOMBRE_ARCHIVO_SALIDA
plt.savefig(ruta_salida, dpi=300, bbox_inches="tight", facecolor='white')
plt.close()

print(f"Gráfico completado con éxito. Guardado en: {ruta_salida}")