"""
================================================================================
VISUALIZACION DE ACTIVIDAD DE TRADING — SEÑALES DE COMPRA Y VENTA
================================================================================
Este script genera visualizaciones para estudiar cuándo el agente (GA o RF)
ha comprado y vendido cada activo de la cartera, cruzando la composición
histórica de pesos con los precios de cierre de cada ticker.

Gráficos generados:
    1. Panel de precios con señales BUY/SELL por ticker (un subplot por activo)
    2. Heatmap de pesos a lo largo del tiempo (todos los tickers × fechas)
    3. Gráfico de actividad de trading: número de operaciones por ticker
    4. Evolución de la cartera con zonas de entrada/salida marcadas

Lógica de detección de señales:
    - COMPRA (BUY) : el peso pasa de 0 (o muy bajo) a positivo entre dos fechas
    - VENTA (SELL) : el peso pasa de positivo a 0 (o muy bajo) entre dos fechas
    - AUMENTO      : el peso aumenta > umbral sin llegar a 0
    - REDUCCION    : el peso disminuye > umbral sin llegar a 0

Autor: Ricardo España Lopez
Fecha: Mayo 2026
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
# 1. CONFIGURACION
# ==============================================================================

ID_EXPERIMENTO  = "GA_MultiStock_v29"
TIPO_MODELO     = "GA"   # "GA" o "RF" (solo afecta al titulo de los graficos)

RUTA_EXPERIMENTO = f"GA_Env/experimentos/{ID_EXPERIMENTO}"
RUTA_BACKTEST    = f"{RUTA_EXPERIMENTO}/backtest_results"
DIR_SALIDA       = f"{RUTA_EXPERIMENTO}/visualizacion_trades"
os.makedirs(DIR_SALIDA, exist_ok=True)

RUTA_COMPOSICION = f"{RUTA_BACKTEST}/portfolio_composition_history.csv"
RUTA_EVOLUCION   = f"{RUTA_BACKTEST}/portfolio_evolution.csv"
RUTA_DATOS       = "GA_Env/data/dataset_GA_estandarizado_v4.csv"

# Umbral para considerar que una posicion ha "cambiado" de forma significativa
# Cambios menores a este valor se ignoran para reducir ruido visual
UMBRAL_CAMBIO_PESO = 0.02   # 1 punto porcentual

# Peso mínimo para considerar que hay posición abierta
UMBRAL_POSICION = 0.005

# Numero maximo de tickers por pagina en el panel de precios
TICKERS_POR_PAGINA = 4

# Paleta de colores
COLOR_BUY      = '#2ca02c'   # verde
COLOR_SELL     = '#d62728'   # rojo
COLOR_AUMENTO  = '#17becf'   # azul claro
COLOR_REDUCCION = '#ff7f0e'  # naranja
COLOR_PRECIO   = '#1f77b4'   # azul oscuro
COLOR_PESO     = '#9467bd'   # violeta


# ==============================================================================
# 2. CARGA DE ARCHIVOS
# ==============================================================================

print("[ BLOQUE 2 ] Cargando archivos de entrada...")

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

# Cargar evolucion si existe (para el grafico 4)
df_evolucion = None
if os.path.exists(RUTA_EVOLUCION):
    df_evolucion = pd.read_csv(RUTA_EVOLUCION)
    df_evolucion['Date'] = pd.to_datetime(df_evolucion['Date'])
    df_evolucion = df_evolucion.sort_values('Date').reset_index(drop=True)
    print(f"  [OK] Evolucion cartera cargada.")

# Tickers activos (excluimos CASH y columnas no-ticker)
tickers = [c for c in df_composicion.columns
           if c not in ['Date', 'CASH'] and df_composicion[c].sum() > 0]

print(f"\n  Periodo   : {df_composicion['Date'].min().date()} a {df_composicion['Date'].max().date()}")
print(f"  Fechas    : {len(df_composicion):,}")
print(f"  Tickers   : {len(tickers)} — {tickers}")
print()


# ==============================================================================
# 3. DETECCION DE SEÑALES DE COMPRA Y VENTA
# ==============================================================================
# Para cada ticker, comparamos el peso en t con el peso en t-1.
# Clasificamos cada transicion en: BUY, SELL, INCREASE, REDUCE, HOLD.

print("[ BLOQUE 3 ] Detectando señales de trading...")

def detectar_señales(df_comp, ticker, umbral_cambio, umbral_posicion):
    """
    Devuelve un DataFrame con las señales de trading para un ticker:
      Date, Peso, Señal (BUY / SELL / INCREASE / REDUCE / HOLD)
    """
    serie = df_comp[['Date', ticker]].copy()
    serie.columns = ['Date', 'Peso']
    serie['Peso_ant'] = serie['Peso'].shift(1).fillna(0)
    serie['Delta'] = serie['Peso'] - serie['Peso_ant']

    condiciones = [
        # Apertura de posicion nueva
        (serie['Peso_ant'] <= umbral_posicion) & (serie['Peso'] > umbral_posicion),
        # Cierre de posicion
        (serie['Peso_ant'] > umbral_posicion) & (serie['Peso'] <= umbral_posicion),
        # Aumento significativo de posicion existente
        (serie['Peso_ant'] > umbral_posicion) & (serie['Peso'] > umbral_posicion)
        & (serie['Delta'] > umbral_cambio),
        # Reduccion significativa de posicion existente
        (serie['Peso_ant'] > umbral_posicion) & (serie['Peso'] > umbral_posicion)
        & (serie['Delta'] < -umbral_cambio),
    ]
    opciones = ['BUY', 'SELL', 'INCREASE', 'REDUCE']
    serie['Señal'] = np.select(condiciones, opciones, default='HOLD')

    return serie

# Construimos el diccionario de señales y tabla resumen
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
    print(f"  {ticker:<12} BUY={n_buy}  SELL={n_sell}  "
          f"INC={n_increase}  RED={n_reduce}  Dias activo={dias_activo}")

df_resumen = pd.DataFrame(resumen_trades).sort_values('Total_Ops', ascending=False)
print()

# Precios de cierre por ticker (pivot)
df_precios = df_datos.pivot_table(index='Date', columns='Ticker', values='Close')
df_precios.index = pd.to_datetime(df_precios.index)


# ==============================================================================
# 4. GRAFICO 1 — PANEL DE PRECIOS CON SEÑALES POR TICKER
# ==============================================================================
# Un subplot por ticker mostrando: precio de cierre + eje secundario con el
# peso asignado + marcadores de BUY (triangulo verde arriba) y
# SELL (triangulo rojo abajo).

print("[ BLOQUE 4 ] Generando Panel de Precios con Señales...")

# Dividimos en paginas si hay muchos tickers
paginas = [tickers[i:i+TICKERS_POR_PAGINA]
           for i in range(0, len(tickers), TICKERS_POR_PAGINA)]

for n_pag, grupo_tickers in enumerate(paginas):
    n_rows = len(grupo_tickers)
    fig, axes = plt.subplots(n_rows, 1, figsize=(15, 4 * n_rows),
                              sharex=True)
    fig.patch.set_facecolor('#f9f9f9')

    if n_rows == 1:
        axes = [axes]

    for ax_idx, ticker in enumerate(grupo_tickers):
        ax1 = axes[ax_idx]
        ax1.set_facecolor('#f9f9f9')

        df_señ = señales_por_ticker[ticker]

        # --- Precio de cierre ---
        if ticker in df_precios.columns:
            precios_ticker = df_precios[ticker].reindex(
                df_señ['Date']).fillna(method='ffill')
            ax1.plot(df_señ['Date'], precios_ticker,
                     color=COLOR_PRECIO, linewidth=1.4,
                     label='Precio cierre', zorder=2)
            ax1.set_ylabel(f'{ticker}\nPrecio (EUR)', fontsize=9,
                           color=COLOR_PRECIO)
            ax1.tick_params(axis='y', labelcolor=COLOR_PRECIO, labelsize=8)

        # --- Zona de posicion activa (fondo sombreado) ---
        en_posicion = False
        fecha_entrada = None
        for _, row in df_señ.iterrows():
            if row['Señal'] == 'BUY' and not en_posicion:
                fecha_entrada = row['Date']
                en_posicion = True
            elif row['Señal'] == 'SELL' and en_posicion:
                ax1.axvspan(fecha_entrada, row['Date'],
                            alpha=0.07, color=COLOR_BUY, zorder=1)
                en_posicion = False
        # Si sigue en posicion al final del periodo
        if en_posicion and fecha_entrada is not None:
            ax1.axvspan(fecha_entrada, df_señ['Date'].iloc[-1],
                        alpha=0.07, color=COLOR_BUY, zorder=1)

        # --- Señales BUY y SELL sobre el precio ---
        if ticker in df_precios.columns:
            df_buy  = df_señ[df_señ['Señal'] == 'BUY']
            df_sell = df_señ[df_señ['Señal'] == 'SELL']
            df_inc  = df_señ[df_señ['Señal'] == 'INCREASE']
            df_red  = df_señ[df_señ['Señal'] == 'REDUCE']

            precio_buy  = precios_ticker.reindex(df_buy['Date'])
            precio_sell = precios_ticker.reindex(df_sell['Date'])
            precio_inc  = precios_ticker.reindex(df_inc['Date'])
            precio_red  = precios_ticker.reindex(df_red['Date'])

            ax1.scatter(df_buy['Date'], precio_buy * 0.985,
                        marker='^', color=COLOR_BUY, s=90, zorder=5,
                        label='BUY')
            ax1.scatter(df_sell['Date'], precio_sell * 1.015,
                        marker='v', color=COLOR_SELL, s=90, zorder=5,
                        label='SELL')
            ax1.scatter(df_inc['Date'], precio_inc * 0.985,
                        marker='^', color=COLOR_AUMENTO, s=50, zorder=4,
                        alpha=0.7, label='INCREASE')
            ax1.scatter(df_red['Date'], precio_red * 1.015,
                        marker='v', color=COLOR_REDUCCION, s=50, zorder=4,
                        alpha=0.7, label='REDUCE')

        # --- Eje secundario: peso asignado ---
        ax2 = ax1.twinx()
        ax2.fill_between(df_señ['Date'], df_señ['Peso'] * 100,
                         alpha=0.20, color=COLOR_PESO, zorder=1)
        ax2.plot(df_señ['Date'], df_señ['Peso'] * 100,
                 color=COLOR_PESO, linewidth=0.8, alpha=0.6)
        ax2.set_ylabel('Peso (%)', fontsize=8, color=COLOR_PESO)
        ax2.tick_params(axis='y', labelcolor=COLOR_PESO, labelsize=7)
        ax2.set_ylim(0, df_señ['Peso'].max() * 100 * 3)

        # Leyenda solo en el primero
        if ax_idx == 0:
            handles = [
                mpatches.Patch(color=COLOR_BUY,      label='BUY (apertura)'),
                mpatches.Patch(color=COLOR_SELL,     label='SELL (cierre)'),
                mpatches.Patch(color=COLOR_AUMENTO,  label='INCREASE (aumento peso)'),
                mpatches.Patch(color=COLOR_REDUCCION,label='REDUCE (reduccion peso)'),
                mpatches.Patch(color=COLOR_PESO,     label='Peso asignado (eje der.)'),
            ]
            ax1.legend(handles=handles, loc='upper left',
                       fontsize=7.5, framealpha=0.85, ncol=2)

        ax1.grid(True, linestyle='--', alpha=0.35, zorder=0)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.suptitle(
        f'Señales de Trading por Ticker — {TIPO_MODELO} ({ID_EXPERIMENTO})\n'
        f'Pagina {n_pag + 1} de {len(paginas)}  |  '
        f'{df_composicion["Date"].min().date()} a {df_composicion["Date"].max().date()}',
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.xticks(rotation=30, fontsize=8)
    plt.tight_layout()

    ruta_panel = os.path.join(
        DIR_SALIDA, f"panel_señales_p{n_pag+1}_{TIPO_MODELO}_{ID_EXPERIMENTO}.png")
    plt.savefig(ruta_panel, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Panel pagina {n_pag+1} guardado: {ruta_panel}")

print()


# ==============================================================================
# 5. GRAFICO 2 — HEATMAP DE PESOS (tickers × tiempo)
# ==============================================================================
# Muestra la evolucion de todos los pesos simultaneamente. Cada fila es un
# ticker; cada columna es una fecha. El color indica el peso asignado.
# Permite identificar de un vistazo las rotaciones de cartera.

print("[ BLOQUE 5 ] Generando Heatmap de pesos...")

df_heatmap = df_composicion.set_index('Date')[tickers].T * 100
df_heatmap = df_heatmap.sort_values(
    df_heatmap.columns[-1], ascending=False)

# Reducir densidad de columnas para legibilidad (muestreo semanal)
fechas_sample = df_heatmap.columns[::max(1, len(df_heatmap.columns) // 60)]
df_heatmap_sample = df_heatmap[fechas_sample]

fig, ax = plt.subplots(figsize=(16, max(6, len(tickers) * 0.6)))
fig.patch.set_facecolor('#f9f9f9')

cmap_custom = LinearSegmentedColormap.from_list(
    'peso_cmap', ['#f0f0f0', '#c6dbef', '#4292c6', '#08306b'])

im = ax.imshow(df_heatmap_sample.values, aspect='auto',
               cmap=cmap_custom, interpolation='nearest',
               vmin=0, vmax=df_heatmap_sample.values.max())

# Etiquetas
ax.set_yticks(range(len(df_heatmap_sample.index)))
ax.set_yticklabels(df_heatmap_sample.index, fontsize=9)

# Eje X con fechas
paso_eje = max(1, len(fechas_sample) // 12)
ax.set_xticks(range(0, len(fechas_sample), paso_eje))
ax.set_xticklabels(
    [f.strftime('%b %y') for f in fechas_sample[::paso_eje]],
    rotation=45, fontsize=8, ha='right')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label('Peso asignado (%)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_title(
    f'Heatmap de Composicion de Cartera — {TIPO_MODELO} ({ID_EXPERIMENTO})\n'
    f'Color = peso asignado (%). Blanco = sin posicion.',
    fontsize=12, fontweight='bold', pad=10
)
ax.set_xlabel('Fecha', fontsize=10)
ax.set_ylabel('Ticker', fontsize=10)

plt.tight_layout()
ruta_heatmap = os.path.join(
    DIR_SALIDA, f"heatmap_pesos_{TIPO_MODELO}_{ID_EXPERIMENTO}.png")
plt.savefig(ruta_heatmap, dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> Heatmap guardado: {ruta_heatmap}")
print()


# ==============================================================================
# 6. GRAFICO 3 — ACTIVIDAD DE TRADING POR TICKER (barras apiladas)
# ==============================================================================
# Muestra cuantas veces el agente ha operado con cada ticker y de qué tipo
# fue la operacion. Permite identificar los activos mas activos en la cartera.

print("[ BLOQUE 6 ] Generando grafico de actividad de trading...")

df_resumen_sorted = df_resumen.sort_values('Total_Ops', ascending=True)

fig, ax = plt.subplots(figsize=(10, max(5, len(tickers) * 0.55)))
fig.patch.set_facecolor('#f9f9f9')
ax.set_facecolor('#f9f9f9')

y_pos = range(len(df_resumen_sorted))
izq = np.zeros(len(df_resumen_sorted))

for col, color, etiq in [
    ('BUY',      COLOR_BUY,      'BUY'),
    ('SELL',     COLOR_SELL,     'SELL'),
    ('INCREASE', COLOR_AUMENTO,  'INCREASE'),
    ('REDUCE',   COLOR_REDUCCION,'REDUCE'),
]:
    vals = df_resumen_sorted[col].values
    ax.barh(list(y_pos), vals, left=izq, color=color,
            alpha=0.85, label=etiq, height=0.65)
    # Anotaciones dentro de las barras
    for i, (v, l) in enumerate(zip(vals, izq)):
        if v > 0:
            ax.text(l + v / 2, i, str(int(v)),
                    ha='center', va='center', fontsize=7.5,
                    color='white', fontweight='bold')
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
ruta_actividad = os.path.join(
    DIR_SALIDA, f"actividad_trading_{TIPO_MODELO}_{ID_EXPERIMENTO}.png")
plt.savefig(ruta_actividad, dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> Actividad de trading guardada: {ruta_actividad}")
print()


# ==============================================================================
# 7. GRAFICO 4 — CURVA DE CAPITAL CON EVENTOS DE REBALANCEO
# ==============================================================================
# Si existe portfolio_evolution.csv, superpone la curva de capital con
# lineas verticales en cada fecha de rebalanceo (cambio de composicion).
# Permite correlacionar visualmente las decisiones de trading con el P&L.

if df_evolucion is not None:
    print("[ BLOQUE 7 ] Generando curva de capital con eventos de rebalanceo...")

    # Detectar columna de valor neto
    posibles_col = ['Net_Worth', 'Portfolio_Value', 'Value',
                    'net_worth', 'portfolio_value']
    col_valor = next((c for c in posibles_col if c in df_evolucion.columns),
                     df_evolucion.select_dtypes(include=[np.number]).columns[0])

    # Detectar fechas de rebalanceo (cualquier cambio en la composicion)
    df_delta = df_composicion[tickers].diff().abs()
    fechas_rebalanceo = df_composicion['Date'][
        df_delta.sum(axis=1) > UMBRAL_CAMBIO_PESO * len(tickers)].values

    # Clasificar rebalanceo por intensidad (numero de tickers cambiados)
    n_tickers_cambiados = (df_delta > UMBRAL_CAMBIO_PESO).sum(axis=1)
    n_tickers_cambiados.index = df_composicion['Date']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)
    fig.patch.set_facecolor('#f9f9f9')
    ax1.set_facecolor('#f9f9f9')
    ax2.set_facecolor('#f9f9f9')

    # --- Panel superior: curva de capital ---
    ax1.plot(df_evolucion['Date'], df_evolucion[col_valor],
             color=COLOR_PRECIO, linewidth=1.8, zorder=3, label='Valor cartera')
    ax1.fill_between(df_evolucion['Date'], df_evolucion[col_valor],
                     df_evolucion[col_valor].min(),
                     alpha=0.07, color=COLOR_PRECIO)

    # Lineas de rebalanceo coloreadas por intensidad
    for fecha in fechas_rebalanceo:
        intensidad = n_tickers_cambiados.get(fecha, 1)
        alpha_val  = min(0.15 + intensidad * 0.06, 0.7)
        ax1.axvline(x=fecha, color='#7f7f7f',
                    linewidth=0.7, alpha=alpha_val,
                    linestyle='--', zorder=2)

    # Marcar maximo y minimo de la curva
    idx_max = df_evolucion[col_valor].idxmax()
    idx_min = df_evolucion[col_valor].idxmin()
    ax1.scatter(df_evolucion.loc[idx_max, 'Date'],
                df_evolucion.loc[idx_max, col_valor],
                color='gold', s=120, zorder=6, marker='*',
                label=f'Maximo: {df_evolucion.loc[idx_max, col_valor]:,.0f} EUR')
    ax1.scatter(df_evolucion.loc[idx_min, 'Date'],
                df_evolucion.loc[idx_min, col_valor],
                color=COLOR_SELL, s=80, zorder=6, marker='X',
                label=f'Minimo: {df_evolucion.loc[idx_min, col_valor]:,.0f} EUR')

    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax1.set_ylabel('Valor de la cartera (EUR)', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.35)
    ax1.set_title(
        f'Curva de Capital con Eventos de Rebalanceo — '
        f'{TIPO_MODELO} ({ID_EXPERIMENTO})\n'
        f'Lineas grises = rebalanceos. Intensidad proporcional al numero de tickers cambiados.',
        fontsize=11, fontweight='bold', pad=10
    )

    # --- Panel inferior: numero de tickers cambiados por fecha ---
    fechas_reb_series = pd.Series(
        n_tickers_cambiados[n_tickers_cambiados > 0])
    ax2.bar(fechas_reb_series.index, fechas_reb_series.values,
            color='#7f7f7f', alpha=0.6, width=3)
    ax2.set_ylabel('N. tickers\ncambiados', fontsize=8)
    ax2.set_xlabel('Fecha', fontsize=10)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, fontsize=8)

    plt.tight_layout()
    ruta_capital = os.path.join(
        DIR_SALIDA, f"capital_rebalanceos_{TIPO_MODELO}_{ID_EXPERIMENTO}.png")
    plt.savefig(ruta_capital, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Curva de capital guardada: {ruta_capital}")
    print()


# ==============================================================================
# 8. EXPORTACION DEL RESUMEN DE OPERACIONES EN .TXT
# ==============================================================================

ruta_resumen_txt = os.path.join(
    DIR_SALIDA, f"resumen_operaciones_{TIPO_MODELO}_{ID_EXPERIMENTO}.txt")

with open(ruta_resumen_txt, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("RESUMEN DE OPERACIONES POR TICKER\n")
    f.write(f"Experimento : {ID_EXPERIMENTO}  |  Modelo: {TIPO_MODELO}\n")
    f.write(f"Generado    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Periodo     : {df_composicion['Date'].min().date()} a "
            f"{df_composicion['Date'].max().date()}\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"{'Ticker':<12} {'BUY':>5} {'SELL':>5} {'INC':>5} "
            f"{'RED':>5} {'TOTAL':>6} {'DIAS_ACTIVO':>12}\n")
    f.write("-" * 55 + "\n")
    for _, row in df_resumen.iterrows():
        f.write(f"{row['Ticker']:<12} {row['BUY']:>5} {row['SELL']:>5} "
                f"{row['INCREASE']:>5} {row['REDUCE']:>5} "
                f"{row['Total_Ops']:>6} {row['Dias_Activo']:>12}\n")

    total_ops = df_resumen['Total_Ops'].sum()
    total_buy  = df_resumen['BUY'].sum()
    total_sell = df_resumen['SELL'].sum()
    f.write("-" * 55 + "\n")
    f.write(f"{'TOTAL':<12} {total_buy:>5} {total_sell:>5} "
            f"{df_resumen['INCREASE'].sum():>5} "
            f"{df_resumen['REDUCE'].sum():>5} {total_ops:>6}\n\n")

    f.write("DETALLE DE SEÑALES POR TICKER Y FECHA\n")
    f.write("-" * 70 + "\n")
    for ticker in tickers:
        df_señ = señales_por_ticker[ticker]
        ops = df_señ[df_señ['Señal'] != 'HOLD'][['Date', 'Señal', 'Peso']]
        if len(ops) > 0:
            f.write(f"\n  {ticker}\n")
            for _, op in ops.iterrows():
                f.write(f"    {op['Date'].date()}  {op['Señal']:<8}  "
                        f"Peso: {op['Peso']*100:.2f}%\n")

    f.write("\n" + "=" * 70 + "\n")

print(f"[ BLOQUE 8 ] Resumen exportado: {ruta_resumen_txt}")
print()


# ==============================================================================
# 9. RESUMEN FINAL
# ==============================================================================
print("=" * 70)
print("VISUALIZACION DE TRADING COMPLETADA")
print("=" * 70)
print(f"Experimento : {ID_EXPERIMENTO}  |  Modelo: {TIPO_MODELO}")
print(f"Tickers analizados : {len(tickers)}")
print(f"Total operaciones  : {df_resumen['Total_Ops'].sum()}")
print()
print("Archivos generados en:", DIR_SALIDA)
for n_pag in range(len(paginas)):
    print(f"  [GRAFICO] panel_señales_p{n_pag+1}_{TIPO_MODELO}_{ID_EXPERIMENTO}.png")
print(f"  [GRAFICO] heatmap_pesos_{TIPO_MODELO}_{ID_EXPERIMENTO}.png")
print(f"  [GRAFICO] actividad_trading_{TIPO_MODELO}_{ID_EXPERIMENTO}.png")
if df_evolucion is not None:
    print(f"  [GRAFICO] capital_rebalanceos_{TIPO_MODELO}_{ID_EXPERIMENTO}.png")
print(f"  [TXT]     resumen_operaciones_{TIPO_MODELO}_{ID_EXPERIMENTO}.txt")
print("=" * 70)
