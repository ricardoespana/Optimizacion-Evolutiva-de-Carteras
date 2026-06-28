# -*- coding: utf-8 -*-
"""
recalculate_backtest.py

Script independiente y completo para regenerar todo el análisis del backtest 
(métricas, gráficos, correlación, sectores y cruce de carteras) leyendo 
directamente los CSV generados por el algoritmo genético.

Incluye todas las correcciones matemáticas:
- Rentabilidad anualizada basada en días naturales reales.
- Volatilidad muestral (ddof=1).
- Supresión del Equal Weight y métricas redundantes.
- Nomenclatura en español para el reporte final.

Autor: Ricardo España Lopez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="notebook")

# ============================================================
# 1. CONFIGURACIÓN Y RUTAS
# ============================================================
ID_EXPERIMENTO = "GA_MultiStock_v28"

DIR_BASE         = Path("GA_Env")
DIR_DATOS        = DIR_BASE / "data"
DIR_EXPERIMENTOS = DIR_BASE / "experimentos"
DIR_SALIDA       = DIR_EXPERIMENTOS / ID_EXPERIMENTO
BACKTEST_DIR     = DIR_SALIDA / "backtest_results"
DIR_CORREGIDO    = BACKTEST_DIR / "resultados_corregidos"

DIR_CORREGIDO.mkdir(parents=True, exist_ok=True)

# Archivos de entrada
RUTA_DATASET     = DIR_DATOS / "dataset_GA_estandarizado_tecnicos.csv"
RUTA_INDICES     = DIR_DATOS / "indices_cierre.csv"
RUTA_PORTFOLIO   = DIR_BASE / "portfolio" / "portfolio_inicial.csv"
RUTA_GA_DETAILS  = BACKTEST_DIR / "ga_path_details.csv"
RUTA_WEIGHTS     = BACKTEST_DIR / "portfolio_composition_history.csv"

# Parámetros financieros
CAPITAL_INICIAL  = 100_000.0
TARGET_PRICE_COL = "Close"
ANIO_INICIO_BT   = 2024
ANIO_FIN_BT      = 2025

# ============================================================
# 2. FUNCIONES DE MÉTRICAS CORREGIDAS
# ============================================================

def calc_corrected_metrics(date_series, capital_series, rf_annual=0.02):
    s = pd.DataFrame({
        "Date"   : pd.to_datetime(np.asarray(date_series)),
        "Capital": np.asarray(capital_series, dtype=float),
    }).dropna().copy()
    s = s.sort_values("Date").drop_duplicates("Date")
    s["returns"] = s["Capital"].pct_change().fillna(0.0)

    if len(s) < 2:
        return {k: np.nan for k in [
            "Rentabilidad total (%)", "Rentabilidad Anualizada (%)", 
            "Volatilidad (%)", "Ratio Sharpe", "Max Drawdown (%)", 
            "Tasa de Acierto (%)", "Valor Final (EUR)"
        ]}

    # CORRECCIÓN: Años reales basados en días naturales
    dias_naturales = (s["Date"].iloc[-1] - s["Date"].iloc[0]).days
    years = dias_naturales / 365.25 if dias_naturales > 0 else np.nan

    total_return = ((s["Capital"].iloc[-1] / s["Capital"].iloc[0]) - 1) * 100
    annualized_return = (
        ((s["Capital"].iloc[-1] / s["Capital"].iloc[0]) ** (1 / years) - 1) * 100
        if pd.notnull(years) and years > 0 else np.nan
    )
    
    volatility = s["returns"].std(ddof=1) * np.sqrt(252) * 100
    
    rf_daily = (1 + rf_annual) ** (1 / 365.25) - 1
    excess_returns = s["returns"] - rf_daily
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / (s["returns"].std(ddof=1) + 1e-12)
    
    cumulative = (1 + s["returns"]).cumprod()
    running_max = cumulative.cummax()
    max_drawdown = (cumulative / running_max - 1).min() * 100
    
    win_rate = (s["returns"] > 0).sum() / len(s["returns"]) * 100

    return {
        "Rentabilidad total (%)"      : total_return,
        "Rentabilidad Anualizada (%)" : annualized_return,
        "Volatilidad (%)"             : volatility,
        "Ratio Sharpe"                : sharpe_ratio,
        "Max Drawdown (%)"            : max_drawdown,
        "Tasa de Acierto (%)"         : win_rate,
        "Valor Final (EUR)"           : s["Capital"].iloc[-1],
    }

# ============================================================
# 3. CARGA DE DATOS Y BENCHMARKS
# ============================================================
print("="*70)
print("INICIANDO RECONSTRUCCIÓN DEL BACKTEST...")
print("="*70)

# Cargar salidas del GA
df_ga = pd.read_csv(RUTA_GA_DETAILS)
df_ga['Date'] = pd.to_datetime(df_ga['Date'])
fechas_bt = df_ga['Date'].values

df_weights = pd.read_csv(RUTA_WEIGHTS)
df_weights['Date'] = pd.to_datetime(df_weights['Date'])

# Cargar mercado y filtrar periodo backtest
df_mercado = pd.read_csv(RUTA_DATASET)
df_mercado['Date'] = pd.to_datetime(df_mercado['Date'])
df_mercado_bt = df_mercado[(df_mercado['Date'].dt.year >= ANIO_INICIO_BT) & 
                           (df_mercado['Date'].dt.year <= ANIO_FIN_BT)].copy()

df_indices = pd.read_csv(RUTA_INDICES)
df_indices['Date'] = pd.to_datetime(df_indices['Date'])

# Recalcular Buy & Hold
print("-> Reconstruyendo Benchmarks...")
df_prices = df_mercado.pivot(index="Date", columns="Ticker", values=TARGET_PRICE_COL).sort_index()
df_prices_bt = df_prices.reindex(df_prices.index.union(pd.to_datetime(fechas_bt))).sort_index().ffill().reindex(pd.to_datetime(fechas_bt)).dropna(axis=1, how="any")
shares_bought = (CAPITAL_INICIAL / df_prices_bt.shape[1]) / df_prices_bt.iloc[0].values
bh_series = (df_prices_bt.values * shares_bought).sum(axis=1)

# Recalcular Índices
df_idx_pivot = df_indices.drop_duplicates(subset=["Date", "Ticker"], keep="last").pivot(index="Date", columns="Ticker", values="Close").sort_index()
df_idx_bt = df_idx_pivot.reindex(df_idx_pivot.index.union(pd.to_datetime(fechas_bt))).sort_index().ffill().reindex(pd.to_datetime(fechas_bt)).dropna(axis=1, how="any")
idx_series = {}
for col in df_idx_bt.columns:
    ret = df_idx_bt[col].pct_change().fillna(0.0)
    nw = CAPITAL_INICIAL * (1 + ret).cumprod()
    nw.iloc[0] = CAPITAL_INICIAL
    idx_series[col] = nw.values

# Ensamblar DataFrame Comparativo
df_comparativo = pd.DataFrame({"Date": pd.to_datetime(fechas_bt)})
df_comparativo["Agente AG"] = df_ga["GA Agent"].values # Traducido desde la raíz
df_comparativo["Buy & Hold"] = bh_series
for k, v in idx_series.items():
    # Aceptamos ambas nomenclaturas (con espacio y con guion bajo) para evitar fallos
    if k in ["IBEX 35", "IBEX_35", "IBEX_MEDIUM_CAP", "IBEX_SMALL_CAP"]:
        df_comparativo[k.replace("_", " ")] = v
        
# DataFrame Normalizado (Base 100)
df_norm = df_comparativo.copy()
for c in [x for x in df_norm.columns if x != "Date"]:
    df_norm[c] = (df_norm[c] / df_norm[c].iloc[0]) * 100

# ============================================================
# 4. CÁLCULO DE MÉTRICAS CORREGIDAS
# ============================================================
print("-> Calculando Métricas Finales...")
resultados_metricas = {}
for c in [x for x in df_comparativo.columns if x != "Date"]:
    resultados_metricas[c] = calc_corrected_metrics(df_comparativo["Date"].values, df_comparativo[c].values)

df_metricas = pd.DataFrame(resultados_metricas).T

# Formateo para la imagen
df_metricas_fmt = df_metricas.copy()
pct_cols = ["Rentabilidad total (%)", "Rentabilidad Anualizada (%)", "Volatilidad (%)", "Max Drawdown (%)", "Tasa de Acierto (%)"]
for c in pct_cols:
    df_metricas_fmt[c] = df_metricas_fmt[c].apply(lambda x: f"{x:.2f}%")
df_metricas_fmt["Ratio Sharpe"] = df_metricas_fmt["Ratio Sharpe"].apply(lambda x: f"{x:.3f}")
df_metricas_fmt["Valor Final (EUR)"] = df_metricas_fmt["Valor Final (EUR)"].apply(lambda x: f"{x:,.2f}")

# Exportar métricas numéricas
df_metricas.to_csv(DIR_CORREGIDO / "metricas_backtest.csv")

# ============================================================
# 5. GENERACIÓN DE GRÁFICOS Y ANÁLISIS
# ============================================================
print("-> Generando visualizaciones y análisis estructurales...")

# 5.1 Tabla de Métricas
fig, ax = plt.subplots(figsize=(14, len(df_metricas_fmt) * 0.55 + 1.5))
ax.axis("off")
col_labels, row_labels = list(df_metricas_fmt.columns), list(df_metricas_fmt.index)
tbl = ax.table(cellText=df_metricas_fmt.values.tolist(), rowLabels=row_labels, colLabels=col_labels, loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)
if "Agente AG" in row_labels:
    idx_ga = row_labels.index("Agente AG") + 1
    for j in range(len(col_labels)): tbl[(idx_ga, j)].set_facecolor("#d6eaf8")
ax.set_title("Métricas del Backtest 2024–2025", fontsize=14, fontweight="bold", pad=12)
plt.savefig(DIR_CORREGIDO / "tabla_metricas.png", dpi=200, bbox_inches="tight")
plt.close()

# 5.2 Evolución Absoluta
plt.figure(figsize=(16, 7))
for c in [x for x in df_comparativo.columns if x != "Date"]:
    lw, ls = (2.5, "-") if c == "Agente AG" else (1.5, "--")
    plt.plot(df_comparativo["Date"], df_comparativo[c], label=c, linewidth=lw, linestyle=ls)
plt.title("Evolución del Valor de Cartera — Backtest 2024-2025", fontsize=14, fontweight="bold")
plt.legend(fontsize=10, ncol=2); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(DIR_CORREGIDO / "grafico_comparativo.png", dpi=300, bbox_inches="tight")
plt.close()

# 5.3 Evolución Base 100
plt.figure(figsize=(16, 7))
for c in [x for x in df_norm.columns if x != "Date"]:
    lw, ls = (2.5, "-") if c == "Agente AG" else (1.5, "--")
    plt.plot(df_norm["Date"], df_norm[c], label=c, linewidth=lw, linestyle=ls)
plt.axhline(100, color="black", linestyle=":", alpha=0.6)
plt.title("Evolución Normalizada (Base 100) — Backtest 2024-2025", fontsize=14, fontweight="bold")
plt.legend(fontsize=10, ncol=2); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(DIR_CORREGIDO / "grafico_base100.png", dpi=300, bbox_inches="tight")
plt.close()

# 5.4 Turnover y Cash
plt.figure(figsize=(16, 5))
plt.plot(df_ga["Date"], df_ga["Turnover"] * 100, label="Turnover (%)", color="black")
plt.fill_between(df_ga["Date"], 0, df_ga["CashWeight"] * 100, alpha=0.25, color="red", label="Cash Weight (%)")
plt.title("Turnover y peso en Liquidez (CASH) — Backtest 2024-2025", fontsize=13, fontweight="bold")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(DIR_CORREGIDO / "turnover_cash.png", dpi=300, bbox_inches="tight")
plt.close()

# 5.5 Drawdown
fig, ax = plt.subplots(figsize=(16, 5))

# Detección dinámica de los nombres exactos en tu dataframe para evitar errores
col_ga = "Agente GA" if "Agente GA" in df_comparativo.columns else ("Agente AG" if "Agente AG" in df_comparativo.columns else "GA Agent")
col_ibex = "IBEX 35" if "IBEX 35" in df_comparativo.columns else "IBEX_35"

for col, color, lw in [(col_ga, "steelblue", 2), (col_ibex, "tomato", 1.5)]:
    if col in df_comparativo.columns:
        # SOLUCIÓN: Usar pandas para limpiar NaNs (ffill/bfill) y calcular el máximo acumulado
        series = df_comparativo[col].astype(float).ffill().bfill()
        
        # cummax() de pandas maneja perfectamente las series temporales
        running_max = series.cummax()
        dd = (series / running_max - 1) * 100
        
        ax.fill_between(df_comparativo["Date"], dd, 0, alpha=0.25, color=color)
        ax.plot(df_comparativo["Date"], dd, label=col.replace("_", " "), color=color, linewidth=lw)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Drawdown — Agente AG vs IBEX 35", fontsize=13, fontweight="bold")
ax.set_ylabel("Drawdown (%)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(DIR_CORREGIDO / "drawdown.png", dpi=200, bbox_inches="tight")
plt.close()
# 5.6 Distribución de Retornos
fig, ax = plt.subplots(figsize=(10, 5))
for col, color in [("Agente AG", "steelblue"), ("IBEX 35", "tomato")]:
    if col in df_comparativo.columns:
        rets = df_comparativo[col].pct_change().dropna() * 100
        ax.hist(rets, bins=60, alpha=0.45, color=color, label=col, edgecolor="none")
        ax.axvline(rets.mean(), color=color, linestyle="--", linewidth=1.2)
ax.set_title("Distribución de Retornos Diarios", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(DIR_CORREGIDO / "distribucion_retornos.png", dpi=200, bbox_inches="tight")
plt.close()

# 5.7 Composición de la Cartera (Stackplot y Barras Finales)
df_plot = df_weights.set_index("Date")

# Quedarnos estrictamente con las acciones que el algoritmo decidió operar
activos_operados = [c for c in df_plot.columns if c != "CASH" and df_plot[c].max() > 0.001]
df_plot_active = df_plot[activos_operados]

# SOLUCIÓN DE COLORES: Concatenar múltiples paletas de matplotlib para asegurar colores únicos
colores_stack = (plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors)

plt.figure(figsize=(18, 8))
plt.stackplot(
    df_plot_active.index, 
    df_plot_active.T.values, 
    labels=df_plot_active.columns, 
    colors=colores_stack[:len(activos_operados)], # Asigna un color único a cada activo
    alpha=0.85
)

# Ajustar la leyenda para que cuadre dinámicamente según el número de acciones
n_cols = max(4, min(10, len(activos_operados) // 2))
plt.legend(loc="upper left", ncol=n_cols, fontsize=8)
plt.title(f"Evolución Temporal de la Composición ({len(activos_operados)} activos operados en total)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(DIR_CORREGIDO / "portfolio_composition_evolution.png", dpi=300, bbox_inches="tight")
plt.close()

# Barras de composición final
last_row = df_plot.iloc[-1]
last_row = last_row[last_row > 0.01].sort_values(ascending=True)
plt.figure(figsize=(10, 6))
bars = plt.barh(last_row.index, last_row.values * 100, color="steelblue", edgecolor="black", alpha=0.85)
for bar in bars:
    plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.2f}%", va="center")
plt.title("Composición Final de la Cartera", fontsize=30, fontweight="bold")
plt.tight_layout()
plt.savefig(DIR_CORREGIDO / "composicion_final_barras.png", dpi=200, bbox_inches="tight")
plt.close()

# 5.8 Correlación de Cartera Final
asset_cols = [c for c in df_weights.columns if c not in ['Date', 'CASH']]
last_weights = df_weights.iloc[-1][asset_cols]
active_tickers = last_weights[last_weights > 0.005].index.tolist()

if len(active_tickers) >= 2:
    df_prices_bt = df_mercado_bt[df_mercado_bt['Ticker'].isin(active_tickers)].copy()
    df_pivot = df_prices_bt.pivot(index='Date', columns='Ticker', values=TARGET_PRICE_COL).sort_index().ffill()
    corr_matrix = df_pivot.pct_change().dropna().corr()
    
    n = len(active_tickers)
    fig_size = max(10, int(n * 0.6))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True, 
                fmt='.2f',
                cmap='RdBu',
                center=0,
                vmin=-1, 
                vmax=1,
                linewidths=0.4, square=True, ax=ax, cbar_kws={'shrink': 0.7, 'label': 'Correlación de Pearson'})
    ax.set_title(f'Matriz de Correlación — Cartera Final ({len(active_tickers)} activos)', fontsize=20, fontweight='bold', pad=14)
    plt.tight_layout()
    plt.savefig(DIR_CORREGIDO / 'correlacion_cartera_final.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5.9 Análisis Sectorial
if 'Sector' in df_mercado_bt.columns:
    sector_map = df_mercado_bt[['Ticker', 'Sector']].dropna().groupby('Ticker')['Sector'].agg(lambda x: x.mode()[0]).to_dict()
    peso_activos = {t: last_weights[t] for t in asset_cols if last_weights.get(t, 0) > 0.005}
    
    sector_weights = {}
    sin_sector = 0.0
    for ticker, peso in peso_activos.items():
        sector = sector_map.get(ticker, None)
        if sector: sector_weights[sector] = sector_weights.get(sector, 0.0) + peso
        else: sin_sector += peso
    if sin_sector > 0: sector_weights['Sin clasificar'] = sin_sector

    sector_df = pd.DataFrame.from_dict(sector_weights, orient='index', columns=['Peso']).sort_values('Peso', ascending=False)
    sector_df['Peso_pct'] = sector_df['Peso'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab20.colors[:len(sector_df)]
    
    bars = axes[0].barh(sector_df.index, sector_df['Peso_pct'], color=colors, edgecolor='black', alpha=0.85)
    for bar in bars: axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.1f}%', va='center')
    axes[0].set_xlabel('Peso en cartera (%)')
    axes[0].set_title('Composición por Sector', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis(); axes[0].grid(axis='x', alpha=0.3)

    axes[1].pie(sector_df['Peso_pct'], labels=sector_df.index, autopct=lambda p: f'{p:.1f}%' if p > 3 else '', colors=colors, startangle=140, wedgeprops=dict(edgecolor='white', linewidth=1.2))
    axes[1].set_title('Distribución Sectorial', fontsize=12, fontweight='bold')
    plt.suptitle('Análisis Sectorial de la Cartera Final', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(DIR_CORREGIDO / 'composicion_sectorial.png', dpi=300, bbox_inches='tight')
    plt.close()
# 5.10 Cruce Cartera Inicial vs Final GA (Reflejando Activos Intocables y Cerrados)
# ── ANÁLISIS: Cruce estructural (Cartera Inicial vs Operativa del GA) ─────────
if RUTA_PORTFOLIO.exists():
    df_inicial = pd.read_csv(RUTA_PORTFOLIO)
    
    # 1. SOLUCIÓN: FILTRADO ESTRICTO POR FECHA (2021-02-01)
    if 'Date' in df_inicial.columns:
        df_inicial['Date'] = pd.to_datetime(df_inicial['Date'])
        df_inicial = df_inicial[df_inicial['Date'] == '2021-02-01']

    ticker_col = next((c for c in df_inicial.columns if c.lower() in ['ticker', 'ric', 'symbol']), None)
    peso_col = next((c for c in df_inicial.columns if c.lower() in ['peso_inicial', 'weight', 'peso']), None)

    if ticker_col and peso_col:
        df_ini = df_inicial[[ticker_col, peso_col]].copy()
        df_ini.columns = ['Ticker', 'Peso_Inicial']
        df_ini['Peso_Inicial'] = pd.to_numeric(df_ini['Peso_Inicial'], errors='coerce').fillna(0.0)
        
        # 2. CARTERA INICIAL (Exactamente las 10 acciones del 2021-02-01)
        tickers_ini = set(df_ini[df_ini['Peso_Inicial'] > 0]['Ticker'].tolist())
        
        # 3. ACTIVOS OPERADOS (Las que tuvieron peso > 0 en cualquier momento de 2024-2025)
        asset_cols = [c for c in df_weights.columns if c not in ['Date', 'CASH']]
        activos_operados = set([c for c in asset_cols if df_weights[c].max() > 0.001])
        
        # 4. UNIVERSO TOTAL REAL (Extraído de las columnas del propio archivo de pesos para evitar errores)
        universo_total = set(asset_cols) | tickers_ini
        
        # 5. CARTERA FINAL (Último día del backtest en 2025)
        last_weights = df_weights.iloc[-1]
        tickers_fin = {t: last_weights[t] for t in asset_cols if last_weights.get(t, 0) > 0.005}
        tickers_fin_set = set(tickers_fin.keys())
        
        # --- MATEMÁTICA DE CATEGORÍAS EXACTAS ---
        # Mantenidas: De las 10 iniciales, ¿cuántas sobreviven al final?
        mantenidas = tickers_ini & tickers_fin_set
        
        # Incorporadas: Acciones nuevas al final que NO estaban en las 10 iniciales
        incorporadas = tickers_fin_set - tickers_ini
        
        # Cerradas: Acciones (de las 10 iniciales o compradas luego) que se vendieron antes del final
        cerradas = (tickers_ini | activos_operados) - tickers_fin_set
        
        # Ignoradas: Acciones del universo que el agente VIO, pero NUNCA tocó ni estaban al principio
        ignoradas = universo_total - (tickers_ini | activos_operados)

        # --- GENERACIÓN DEL GRÁFICO ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        categorias = ['Mantenidas\n(Desde 2021)', 'Incorporadas\n(Nuevas)', 'Cerradas\n(Vendidas)', 'Ignoradas\n(Nunca tocadas)']
        counts = [len(mantenidas), len(incorporadas), len(cerradas), len(ignoradas)]
        colores = ['#2ecc71', '#3498db', '#e74c3c', '#bdc3c7']
        
        bars = axes[0].bar(categorias, counts, color=colores, edgecolor='black', alpha=0.85, width=0.6)
        for bar, count in zip(bars, counts): 
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        axes[0].set_ylabel('Número de activos')
        axes[0].set_title(f'Gestión del Universo ({len(universo_total)} acciones) durante el backtest', fontsize=12, fontweight='bold')
        axes[0].set_ylim(0, max(counts) + max(5, max(counts)*0.15))
        
        axes[0].text(0.98, 0.97, f'Acciones Iniciales (2021): {len(tickers_ini)}\nActivos Operados: {len(activos_operados)}\nUniverso Total: {len(universo_total)}', transform=axes[0].transAxes, ha='right', va='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        if mantenidas:
            df_mant = pd.DataFrame([{'Ticker': t, 'Inicial': float(df_ini[df_ini['Ticker']==t]['Peso_Inicial'].values[0]), 'Final': tickers_fin[t]} for t in mantenidas]).sort_values('Final')
            x = np.arange(len(df_mant))
            w = 0.35
            axes[1].bar(x - w/2, df_mant['Inicial']*100, w, label='Inicial (2021)', color='#95a5a6', alpha=0.85)
            axes[1].bar(x + w/2, df_mant['Final']*100,   w, label='Final AG (2025)', color='#2ecc71', alpha=0.85)
            axes[1].set_xticks(x); axes[1].set_xticklabels(df_mant['Ticker'], rotation=45, ha='right', fontsize=8)
            axes[1].set_ylabel('Peso en cartera (%)')
            axes[1].set_title('Comparativa de Pesos — Acciones Mantenidas', fontsize=12, fontweight='bold')
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'Ninguna acción mantenida de las 10 iniciales', ha='center', va='center', fontsize=12, color='gray', transform=axes[1].transAxes)
        
        plt.suptitle('Cruce Estructural: Cartera de Origen (2021) vs Cartera del Agente (2025)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(BACKTEST_DIR / 'cruce_inicial_final.png', dpi=300, bbox_inches='tight')
        plt.close()

        # --- Exportar tabla de detalle ---
        rows = []
        for t in sorted(mantenidas):
            p_i = float(df_ini[df_ini['Ticker']==t]['Peso_Inicial'].values[0])
            rows.append({'Ticker': t, 'Estado': 'Mantenida', 'Peso_Ini_%': round(p_i*100, 2), 'Peso_Fin_%': round(tickers_fin[t]*100, 2)})
        for t in sorted(incorporadas):
            rows.append({'Ticker': t, 'Estado': 'Incorporada', 'Peso_Ini_%': 0.0, 'Peso_Fin_%': round(tickers_fin[t]*100, 2)})
        for t in sorted(cerradas):
            p_i = float(df_ini[df_ini['Ticker']==t]['Peso_Inicial'].values[0]) if t in tickers_ini else 0.0
            rows.append({'Ticker': t, 'Estado': 'Cerrada_o_Vendida', 'Peso_Ini_%': round(p_i*100, 2), 'Peso_Fin_%': 0.0})
        
        df_cruce = pd.DataFrame(rows)
        df_cruce.to_csv(BACKTEST_DIR / 'detalle_cruce_carteras.csv', index=False)
        print(f"✔ Gráfico de cruce estructural (corregido a 2021-02-01) guardado en: {BACKTEST_DIR / 'cruce_inicial_final.png'}")