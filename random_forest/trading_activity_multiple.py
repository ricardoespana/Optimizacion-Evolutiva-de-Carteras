"""
================================================================================
VISUALIZACION DE ACTIVIDAD DE TRADING — SEÑALES DE COMPRA Y VENTA
================================================================================
Este script genera visualizaciones para estudiar cuándo el agente (AG o PPO)
ha comprado y vendido cada activo de la cartera, cruzando la composición
histórica de pesos con los precios de cierre de cada ticker.

Gráficos generados:
    1. Panel de precios con señales BUY/SELL por ticker (un subplot por activo)
    2. Heatmap de pesos a lo largo del tiempo (todos los tickers × fechas)
    3. Gráfico de actividad de trading: número de operaciones por ticker
    4. Evolución de la cartera con zonas de entrada/salida marcadas

Autor: Ricardo España Lopez
Fecha: Junio 2026
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

print("=" * 70)
print("VISUALIZACION DE ACTIVIDAD DE TRADING — SEÑALES DE COMPRA/VENTA")
print("=" * 70)
print(f"Fecha de ejecucion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()


# ==============================================================================
# 1. CONFIGURACION Y PARAMETRIZACIÓN DE ENRUTAMIENTO INDEPENDIENTE
# ==============================================================================

ID_EXPERIMENTO  = "GA_MultiStock_v28" #LOS DATOS TÉCNICOS ES A PARTIR DE V_27
MODELO_TEST     = "AG"                # Opciones: "AG" o "RF" RF SOLO SIRVE PARA IBEX
MERCADO_TEST    = "IBEX"              # Opciones: "IBEX" o "S&P_500"

# Mapeo de variable heredada para compatibilidad de títulos
TIPO_MODELO     = MODELO_TEST

RUTA_EXPERIMENTO = f"GA_Env/experimentos/{ID_EXPERIMENTO}"

# Matriz de enrutamiento estricto por experimento concreto
if MODELO_TEST == "AG" and MERCADO_TEST == "IBEX":
    RUTA_BACKTEST    = f"{RUTA_EXPERIMENTO}/backtest_results"
    RUTA_EVOLUCION   = f"{RUTA_BACKTEST}/portfolio_evolution.csv"
    RUTA_DATOS       = "GA_Env/data/dataset_GA_estandarizado_v4.csv"

elif MODELO_TEST == "AG" and MERCADO_TEST == "S&P_500":
    RUTA_BACKTEST    = f"{RUTA_EXPERIMENTO}/backtest_S&P_v2"
    RUTA_EVOLUCION   = f"{RUTA_BACKTEST}/portfolio_evolution.csv"
    RUTA_DATOS       = "GA_Env/data/sp500_estandarizado_backtest_v2.csv"

elif MODELO_TEST == "RF" and MERCADO_TEST == "IBEX":
    RUTA_BACKTEST    = f"{RUTA_EXPERIMENTO}/rf_homogeneo/backtest_results_rf"
    RUTA_EVOLUCION   = f"{RUTA_BACKTEST}/portfolio_evolution.csv"
    RUTA_DATOS       = "GA_Env/data/dataset_GA_estandarizado_v4.csv"

elif MODELO_TEST == "RF" and MERCADO_TEST == "S&P_500":
    RUTA_BACKTEST    = f"{RUTA_EXPERIMENTO}/rf_homogeneo/backtest_results_rf"
    RUTA_EVOLUCION   = f"{RUTA_BACKTEST}/portfolio_evolution.csv"
    RUTA_DATOS       = "GA_Env/data/sp500_estandarizado_backtest.csv"

else:
    raise ValueError("La combinación de MODELO_TEST y MERCADO_TEST configurada no es válida.")

# La carpeta de salida se genera de forma específica dentro del directorio del experimento concreto
DIR_SALIDA = f"{RUTA_BACKTEST}/visualizacion_trades_v2"
os.makedirs(DIR_SALIDA, exist_ok=True)

RUTA_COMPOSICION = f"{RUTA_BACKTEST}/portfolio_composition_history.csv"

# Parámetros del algoritmo visual
UMBRAL_CAMBIO_PESO = 0.02   
UMBRAL_POSICION    = 0.005
TICKERS_POR_PAGINA = 4

# Paleta cromática de la memoria
COLOR_BUY       = '#2ca02c'   
COLOR_SELL      = '#d62728'   
COLOR_AUMENTO   = '#17becf'   
COLOR_REDUCCION = '#ff7f0e'   
COLOR_PRECIO    = '#1f77b4'   
COLOR_PESO      = '#9467bd'   


# ==============================================================================
# 2. CARGA DE ARCHIVOS
# ==============================================================================

print(f"[ BLOQUE 2 ] Cargando archivos para el experimento: {MODELO_TEST} - {MERCADO_TEST}...")

for nombre, ruta in [("Composicion", RUTA_COMPOSICION),
                     ("Datos mercado", RUTA_DATOS)]:
    if not os.path.exists(ruta):
        print(f"  ERROR: No se encuentra '{nombre}': {ruta}")
        sys.exit(1)
    print(f"  [OK] {nombre}: {ruta}")

df_composicion = pd.read_csv(RUTA_COMPOSICION)
df_composicion['Date'] = pd.to_datetime(df_composicion['Date'])
df_composicion = df_composicion.sort_values('Date').reset_index(drop=True)

df_datos = pd.read_csv(RUTA_DATOS)
df_datos['Date'] = pd.to_datetime(df_datos['Date'])

df_evolucion = None
if os.path.exists(RUTA_EVOLUCION):
    df_evolucion = pd.read_csv(RUTA_EVOLUCION)
    df_evolucion['Date'] = pd.to_datetime(df_evolucion['Date'])
    df_evolucion = df_evolucion.sort_values('Date').reset_index(drop=True)
    print(f"  [OK] Evolucion de cartera cargada correctamente.")

# Aislamiento de activos excluyendo CASH y vectores vacíos
tickers = [c for c in df_composicion.columns if c not in ['Date', 'CASH'] and df_composicion[c].sum() > 0]

print(f"\n  Horizonte Temporal : {df_composicion['Date'].min().date()} al {df_composicion['Date'].max().date()}")
print(f"  Muestras Diarias   : {len(df_composicion):,}")
print(f"  Universo Activos   : {len(tickers)} — {tickers}")
print()


# ==============================================================================
# 3. DETECCION DE SEÑALES DE COMPRA Y VENTA
# ==============================================================================

print("[ BLOQUE 3 ] Detectando transiciones y señales operativas...")

def detectar_señales(df_comp, ticker, umbral_cambio, umbral_posicion):
    serie = df_comp[['Date', ticker]].copy()
    serie.columns = ['Date', 'Peso']
    serie['Peso_ant'] = serie['Peso'].shift(1).fillna(0)
    serie['Delta'] = serie['Peso'] - serie['Peso_ant']

    condiciones = [
        (serie['Peso_ant'] <= umbral_posicion) & (serie['Peso'] > umbral_posicion),
        (serie['Peso_ant'] > umbral_posicion) & (serie['Peso'] <= umbral_posicion),
        (serie['Peso_ant'] > umbral_posicion) & (serie['Original_Weight'] if 'Original_Weight' in serie.columns else serie['Peso'] > umbral_posicion) & (serie['Delta'] > umbral_cambio),
        (serie['Peso_ant'] > umbral_posicion) & (serie['Original_Weight'] if 'Original_Weight' in serie.columns else serie['Peso'] > umbral_posicion) & (serie['Delta'] < -umbral_cambio),
    ]
    # Pequeña corrección para asegurar el fallback de condiciones si no existe columna duplicada
    condiciones[2] = (serie['Peso_ant'] > umbral_posicion) & (serie['Peso'] > umbral_posicion) & (serie['Delta'] > umbral_cambio)
    condiciones[3] = (serie['Peso_ant'] > umbral_posicion) & (serie['Peso'] > umbral_posicion) & (serie['Delta'] < -umbral_cambio)

    opciones = ['BUY', 'SELL', 'INCREASE', 'REDUCE']
    serie['Señal'] = np.select(condiciones, opciones, default='HOLD')
    return serie

señales_por_ticker = {}
resumen_trades = []

for ticker in tickers:
    df_señ = detectar_señales(df_composicion, ticker, UMBRAL_CAMBIO_PESO, UMBRAL_POSICION)
    señales_por_ticker[ticker] = df_señ

    n_buy      = (df_señ['Señal'] == 'BUY').sum()
    n_sell     = (df_señ['Señal'] == 'SELL').sum()
    n_increase = (df_señ['Señal'] == 'INCREASE').sum()
    n_reduce   = (df_señ['Señal'] == 'REDUCE').sum()
    dias_activo = (df_señ['Peso'] > UMBRAL_POSICION).sum()

    resumen_trades.append({
        'Ticker'   : ticker,
        'BUY'      : n_buy,
        'SELL'     : n_sell,
        'INCREASE' : n_increase,
        'REDUCE'   : n_reduce,
        'Total_Ops': n_buy + n_sell + n_increase + n_reduce,
        'Dias_Activo': dias_activo,
    })
    print(f"  {ticker:<12} BUY={n_buy}  SELL={n_sell}  INC={n_increase}  RED={n_reduce}  Dias Activo={dias_activo}")

df_resumen = pd.DataFrame(resumen_trades).sort_values('Total_Ops', ascending=False)
print()

df_precios = df_datos.pivot_table(index='Date', columns='Ticker', values='Close')
df_precios.index = pd.to_datetime(df_precios.index)


# ==============================================================================
# 4. GRAFICO 1 — PANEL DE PRECIOS CON SEÑALES POR TICKER
# ==============================================================================

print("[ BLOQUE 4 ] Generando Matrices Gráficas de Precios y Señales...")

paginas = [tickers[i:i+TICKERS_POR_PAGINA] for i in range(0, len(tickers), TICKERS_POR_PAGINA)]

for n_pag, grupo_tickers in enumerate(paginas):
    n_rows = len(grupo_tickers)
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, 4 * n_rows), sharex=True)
    fig.patch.set_facecolor('#f9f9f9')

    if n_rows == 1:
        axes = [axes]

    for ax_idx, ticker in enumerate(grupo_tickers):
        ax1 = axes[ax_idx]
        ax1.set_facecolor('#f9f9f9')
        df_señ = señales_por_ticker[ticker]

        if ticker in df_precios.columns:
            precios_ticker = df_precios[ticker].reindex(df_señ['Date']).fillna(method='ffill')
            ax1.plot(df_señ['Date'], precios_ticker, color=COLOR_PRECIO, linewidth=1.4, label='Precio cierre', zorder=2)
            ax1.set_ylabel(f'{ticker}\nPrecio (EUR)', fontsize=9, color=COLOR_PRECIO)
            ax1.tick_params(axis='y', labelcolor=COLOR_PRECIO, labelsize=8)

        en_posicion = False
        fecha_entrada = None
        for _, row in df_señ.iterrows():
            if row['Señal'] == 'BUY' and not en_posicion:
                fecha_entrada = row['Date']
                en_posicion = True
            elif row['Señal'] == 'SELL' and en_posicion:
                ax1.axvspan(fecha_entrada, row['Date'], alpha=0.07, color=COLOR_BUY, zorder=1)
                en_posicion = False
        
        if en_posicion and fecha_entrada is not None:
            ax1.axvspan(fecha_entrada, df_señ['Date'].iloc[-1], alpha=0.07, color=COLOR_BUY, zorder=1)

        if ticker in df_precios.columns:
            df_buy  = df_señ[df_señ['Señal'] == 'BUY']
            df_sell = df_señ[df_señ['Señal'] == 'SELL']
            df_inc  = df_señ[df_señ['Señal'] == 'INCREASE']
            df_red  = df_señ[df_señ['Señal'] == 'REDUCE']

            ax1.scatter(df_buy['Date'], precios_ticker.reindex(df_buy['Date']) * 0.985, marker='^', color=COLOR_BUY, s=90, zorder=5)
            ax1.scatter(df_sell['Date'], precios_ticker.reindex(df_sell['Date']) * 1.015, marker='v', color=COLOR_SELL, s=90, zorder=5)
            ax1.scatter(df_inc['Date'], precios_ticker.reindex(df_inc['Date']) * 0.985, marker='^', color=COLOR_AUMENTO, s=50, zorder=4, alpha=0.7)
            ax1.scatter(df_red['Date'], precios_ticker.reindex(df_red['Date']) * 1.015, marker='v', color=COLOR_REDUCCION, s=50, zorder=4, alpha=0.7)

        ax2 = ax1.twinx()
        ax2.fill_between(df_señ['Date'], df_señ['Peso'] * 100, alpha=0.20, color=COLOR_PESO, zorder=1)
        ax2.plot(df_señ['Date'], df_señ['Peso'] * 100, color=COLOR_PESO, linewidth=0.8, alpha=0.6)
        ax2.set_ylabel('Peso (%)', fontsize=8, color=COLOR_PESO)
        ax2.tick_params(axis='y', labelcolor=COLOR_PESO, labelsize=7)
        ax2.set_ylim(0, df_señ['Peso'].max() * 100 * 3)

        if ax_idx == 0:
            handles = [
                mpatches.Patch(color=COLOR_BUY,      label='BUY (apertura)'),
                mpatches.Patch(color=COLOR_SELL,     label='SELL (cierre)'),
                mpatches.Patch(color=COLOR_AUMENTO,  label='INCREASE (aumento peso)'),
                mpatches.Patch(color=COLOR_REDUCCION,label='REDUCE (reduccion peso)'),
                mpatches.Patch(color=COLOR_PESO,     label='Peso asignado (eje der.)'),
            ]
            ax1.legend(handles=handles, loc='upper left', fontsize=7.5, framealpha=0.85, ncol=2)

        ax1.grid(True, linestyle='--', alpha=0.35, zorder=0)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.suptitle(
        f'Señales de Trading por Ticker — {TIPO_MODELO} ({ID_EXPERIMENTO})\n'
        f'Pagina {n_pag + 1} de {len(paginas)}  |  Mercado: {MERCADO_TEST}',
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.xticks(rotation=30, fontsize=8)
    plt.tight_layout()

    ruta_panel = os.path.join(DIR_SALIDA, f"panel_señales_p{n_pag+1}_{TIPO_MODELO}_{ID_EXPERIMENTO}.png")
    plt.savefig(ruta_panel, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Panel pagina {n_pag+1} guardado en su directorio: {ruta_panel}")

print()


# ==============================================================================
# 5. GRAFICO 2 — HEATMAP DE PESOS (tickers × tiempo)
# ==============================================================================



import matplotlib.colors as mcolors

print("[ BLOQUE 5 ] Generando Heatmap estructural de rotación...")
plt.style.use('seaborn-v0_8-white') # <--- AÑADE ESTO AQUÍ

df_heatmap = df_composicion.set_index('Date')[tickers].T
df_heatmap = df_heatmap.sort_values(df_heatmap.columns[-1], ascending=False)

fechas_sample = df_heatmap.columns[::max(1, len(df_heatmap.columns) // 60)]
df_heatmap_sample = df_heatmap[fechas_sample]

fig, ax = plt.subplots(figsize=(16, max(6, len(tickers) * 0.6)))
fig.patch.set_facecolor('#f9f9f9')

# Códigos hexadecimales exactos de la mitad positiva de RdBu
colores_rdbu_blue = [
    '#ffffff', # 0.0: Blanco puro (sin posición)
    '#d1e5f0', # Azul muy pálido/grisáceo (migajas residuales)
    '#92c5de', # Azul claro apagado
    '#4393c3', # Azul medio RdBu
    '#2166ac', # Azul oscuro clásico de RdBu
    '#053061'  # 1.0: Azul marino casi negro RdBu
]
cmap_custom = mcolors.LinearSegmentedColormap.from_list('rdbu_blue_cmap', colores_rdbu_blue)
# Mantenemos el PowerNorm para seguir viendo las micro-asignaciones
norm = mcolors.PowerNorm(gamma=0.35, vmin=0.0, vmax=1.0)
# ====================================================================

im = ax.imshow(df_heatmap_sample.values, aspect='auto', cmap=cmap_custom, norm=norm, interpolation='nearest')
# Eje Y (Tickers)
ax.set_yticks(range(len(df_heatmap_sample.index)))
ax.set_yticklabels(df_heatmap_sample.index, fontsize=20)

# Eje X (Fechas)
paso_eje = max(1, len(fechas_sample) // 12)
ax.set_xticks(range(0, len(fechas_sample), paso_eje))
ax.set_xticklabels([f.strftime('%b %y') for f in fechas_sample[::paso_eje]], rotation=45, fontsize=20, ha='right')

# ====================================================================
# CREACIÓN DE LA CUADRÍCULA DELIMITADORA ---
# ====================================================================
# Desplazamos las líneas -0.5 para que caigan justo en los bordes de los bloques
ax.set_xticks(np.arange(-.5, len(df_heatmap_sample.columns), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(df_heatmap_sample.index), 1), minor=True)
# Dibujamos el grid basándonos en esas marcas menores (color gris suave para no saturar)
ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.5, alpha=0.8)
# Ocultamos los "palitos" de las marcas menores para que quede limpio
ax.tick_params(which="minor", bottom=False, left=False)
# ====================================================================

cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label('Peso en Cartera (0 a 1)', fontsize=9, fontweight='bold')
cbar.ax.tick_params(labelsize=8)

ax.set_title(
    f'Heatmap de Composición de Cartera ({len(tickers)} activos operados) — {TIPO_MODELO} {MERCADO_TEST} \n',
    fontsize=30, fontweight='bold', pad=10
)
ax.set_xlabel('Fecha', fontsize=30)
ax.set_ylabel('Ticker', fontsize=30)

plt.tight_layout()
ruta_heatmap = os.path.join(DIR_SALIDA, f"heatmap_pesos_{TIPO_MODELO}_{ID_EXPERIMENTO}.png")
plt.savefig(ruta_heatmap, dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> Heatmap guardado en su directorio: {ruta_heatmap}")


# ==============================================================================
# 6. GRAFICO 3 — ACTIVIDAD DE TRADING POR TICKER (barras apiladas)
# ==============================================================================

print("[ BLOQUE 6 ] Generando histograma de volumen operativo...")

df_resumen_sorted = df_resumen.sort_values('Total_Ops', ascending=True)

# Cambia el antiguo fig.patch.set_facecolor('#f9f9f9') por esto:
fig, ax = plt.subplots(figsize=(16, max(6, len(tickers) * 0.6)), facecolor='white')
fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#f9f9f9')

y_pos = range(len(df_resumen_sorted))
izq = np.zeros(len(df_resumen_sorted))

for col, color, etiq in [('BUY', COLOR_BUY, 'BUY'), ('SELL', COLOR_SELL, 'SELL'), ('INCREASE', COLOR_AUMENTO, 'INCREASE'), ('REDUCE', COLOR_REDUCCION, 'REDUCE')]:
    vals = df_resumen_sorted[col].values
    ax.barh(list(y_pos), vals, left=izq, color=color, alpha=0.85, label=etiq, height=0.65)
    for i, (v, l) in enumerate(zip(vals, izq)):
        if v > 0:
            ax.text(l + v / 2, i, str(int(v)), ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
    izq += vals

ax.set_yticks(list(y_pos))
ax.set_yticklabels(df_resumen_sorted['Ticker'], fontsize=9)
ax.set_xlabel('Numero de operaciones', fontsize=10)
ax.set_title(
    f'Actividad de Trading por Ticker — {TIPO_MODELO} ({ID_EXPERIMENTO})',
    fontsize=12, fontweight='bold', pad=10
)
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.grid(True, axis='x', linestyle='--', alpha=0.4)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.tight_layout()
ruta_actividad = os.path.join(DIR_SALIDA, f"actividad_trading_{TIPO_MODELO}_{ID_EXPERIMENTO}.png")
plt.savefig(ruta_actividad, dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> Gráfico de actividad operativa guardado: {ruta_actividad}")
print()


# ==============================================================================
# 7. GRAFICO 4 — CURVA DE CAPITAL CON EVENTOS DE REBALANCEO
# ==============================================================================

if df_evolucion is not None:
    print("[ BLOQUE 7 ] Generando curva de capital vinculada a los rebalanceos...")

    posibles_col = ['Net_Worth', 'Portfolio_Value', 'Value', 'net_worth', 'portfolio_value', 
                    'PPO_Base100', 'GA_Base100', 'Agente PPO', 'Agente AG', 'GA Agent']
    col_valor = next((c for c in posibles_col if c in df_evolucion.columns),
                     df_evolucion.select_dtypes(include=[np.number]).columns[0])

    df_delta = df_composicion[tickers].diff().abs()
    fechas_rebalanceo = df_composicion['Date'][df_delta.sum(axis=1) > UMBRAL_CAMBIO_PESO * len(tickers)].values
    n_tickers_cambiados = (df_delta > UMBRAL_CAMBIO_PESO).sum(axis=1)
    n_tickers_cambiados.index = df_composicion['Date']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.patch.set_facecolor('#f9f9f9')
    ax1.set_facecolor('#f9f9f9')
    ax2.set_facecolor('#f9f9f9')

    ax1.plot(df_evolucion['Date'], df_evolucion[col_valor], color=COLOR_PRECIO, linewidth=1.8, zorder=3, label='Valor cartera')
    ax1.fill_between(df_evolucion['Date'], df_evolucion[col_valor], df_evolucion[col_valor].min(), alpha=0.07, color=COLOR_PRECIO)

    for fecha in fechas_rebalanceo:
        intensidad = n_tickers_cambiados.get(fecha, 1)
        alpha_val  = min(0.15 + intensidad * 0.06, 0.7)
        ax1.axvline(x=fecha, color='#7f7f7f', linewidth=0.7, alpha=alpha_val, linestyle='--', zorder=2)

    idx_max = df_evolucion[col_valor].idxmax()
    idx_min = df_evolucion[col_valor].idxmin()
    ax1.scatter(df_evolucion.loc[idx_max, 'Date'], df_evolucion.loc[idx_max, col_valor], color='gold', s=120, zorder=6, marker='*', label=f'Maximo: {df_evolucion.loc[idx_max, col_valor]:,.0f}')
    ax1.scatter(df_evolucion.loc[idx_min, 'Date'], df_evolucion.loc[idx_min, col_valor], color=COLOR_SELL, s=80, zorder=6, marker='X', label=f'Minimo: {df_evolucion.loc[idx_min, col_valor]:,.0f}')

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax1.set_ylabel('Valor de la cartera', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.35)
    ax1.set_title(
        f'Curva de Capital con Eventos de Rebalanceo — {TIPO_MODELO} \n'
        f'Lineas grises = Rebalanceo operativo en {MERCADO_TEST}.',
        fontsize=11, fontweight='bold', pad=10
    )

    fechas_reb_series = pd.Series(n_tickers_cambiados[n_tickers_cambiados > 0])
    ax2.bar(fechas_reb_series.index, fechas_reb_series.values, color='#7f7f7f', alpha=0.6, width=3)
    ax2.set_ylabel('Operaciones \nrealizadas', fontsize=8)
    ax2.set_xlabel('Fecha', fontsize=10)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, fontsize=8)

    plt.tight_layout()
    ruta_capital = os.path.join(DIR_SALIDA, f"capital_rebalanceos_{TIPO_MODELO}_{ID_EXPERIMENTO}.png")
    plt.savefig(ruta_capital, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Curva de capital guardada: {ruta_capital}")
    print()

# ==============================================================================
# 8. EXPORTACION DEL RESUMEN DE OPERACIONES EN .TXT
# ==============================================================================

ruta_resumen_txt = os.path.join(DIR_SALIDA, f"resumen_operaciones_{TIPO_MODELO}_{ID_EXPERIMENTO}.txt")

with open(ruta_resumen_txt, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("RESUMEN DE OPERACIONES POR TICKER\n")
    f.write(f"Experimento : {ID_EXPERIMENTO}  |  Modelo: {TIPO_MODELO}  |  Mercado: {MERCADO_TEST}\n")
    f.write(f"Generado    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Periodo     : {df_composicion['Date'].min().date()} a {df_composicion['Date'].max().date()}\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"{'Ticker':<12} {'BUY':>5} {'SELL':>5} {'INC':>5} {'RED':>5} {'TOTAL':>6} {'DIAS_ACTIVO':>12}\n")
    f.write("-" * 55 + "\n")
    for _, row in df_resumen.iterrows():
        f.write(f"{row['Ticker']:<12} {row['BUY']:>5} {row['SELL']:>5} {row['INCREASE']:>5} {row['REDUCE']:>5} {row['Total_Ops']:>6} {row['Dias_Activo']:>12}\n")

    total_ops = df_resumen['Total_Ops'].sum()
    total_buy  = df_resumen['BUY'].sum()
    total_sell = df_resumen['SELL'].sum()
    f.write("-" * 55 + "\n")
    f.write(f"{'TOTAL':<12} {total_buy:>5} {total_sell:>5} {df_resumen['INCREASE'].sum():>5} {df_resumen['REDUCE'].sum():>5} {total_ops:>6}\n\n")

    f.write("DETALLE DE SEÑALES POR TICKER Y FECHA\n")
    f.write("-" * 70 + "\n")
    for ticker in tickers:
        df_señ = señales_por_ticker[ticker]
        ops = df_señ[df_señ['Señal'] != 'HOLD'][['Date', 'Señal', 'Peso']]
        if len(ops) > 0:
            f.write(f"\n  {ticker}\n")
            for _, op in ops.iterrows():
                f.write(f"    {op['Date'].date()}  {op['Señal']:<8}  Peso: {op['Peso']*100:.2f}%\n")

    f.write("\n" + "=" * 70 + "\n")

print(f"[ BLOQUE 8 ] Auditoría resumida exportada en texto: {ruta_resumen_txt}")
print()


# ==============================================================================
# 9. RESUMEN FINAL POR CONSOLA
# ==============================================================================
print("=" * 70)
print("VISUALIZACION DE TRADING COMPLETADA CON ÉXITO")
print("=" * 70)
print(f"Directorio de Salida Aislado: {DIR_SALIDA}")
print(f"Modelo Evaluado : Agente {TIPO_MODELO}  |  Mercado Objetivo: {MERCADO_TEST}")
print(f"Activos con Posesión Abierta: {len(tickers)}")
print(f"Suma de Operaciones Reales   : {df_resumen['Total_Ops'].sum()}")
print("=" * 70)