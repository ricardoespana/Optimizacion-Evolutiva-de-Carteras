import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json

from trading_env_multistock_ga_v3 import MultiStockTradingEnvGA2

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="notebook")

"""
ESTE BACKTEST ES PARA HACERLO ENTRE DIFERENTES DATOS DE CADA ÍNDICE PARA AG
"""
# ============================================================
# 1. CONFIGURACIÓN DE NOMBRES Y RUTAS
# ============================================================
ID_EXPERIMENTO = "GA_MultiStock_v28"

# --- Parametrización de Nombres ---
NOMBRE_INDICE_TEST   = "S&P 500"      
NOMBRE_ENTRENAMIENTO = "Modelo IBEX"  
NOMBRE_TEST          = "Modelo S&P"   

# --- Directorios ---
DIR_BASE         = Path("GA_Env")
DIR_DATOS        = DIR_BASE / "data"
DIR_EXPERIMENTOS = DIR_BASE / "experimentos"
DIR_SALIDA       = DIR_EXPERIMENTOS / ID_EXPERIMENTO

# Carpeta destino para todo lo del S&P 500
BACKTEST_DIR     = DIR_SALIDA / "backtest_S&P_v2" 
DIR_COMPARATIVA  = BACKTEST_DIR / "comparativa_entrenamiento_vs_test"

BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
DIR_COMPARATIVA.mkdir(parents=True, exist_ok=True)

# --- Archivos de entrada ---
RUTA_DATASET     = DIR_DATOS / "sp500_estandarizado_backtest_tecnicos.csv"
RUTA_INDICES     = DIR_DATOS / "cotiz_SPX.csv" 
RUTA_CONFIG      = DIR_SALIDA / "best_config_ga.json"
RUTA_GENOMA      = DIR_SALIDA / "best_genome.npy"
RUTA_PORTFOLIO   = DIR_BASE / "portfolio" / "portfolio_inicial.csv"

# --- Archivo del backtest original (Para la comparativa) ---
RUTA_BACKTEST_ORIGINAL = DIR_SALIDA / "backtest_results" / "portfolio_evolution_base100.csv"

# ============================================================
# 2. CLASES Y FUNCIONES DE SIMULACIÓN Y MÉTRICAS
# ============================================================
class GeneticMLPPolicy:
    def __init__(self, obs_dim, action_dim, hidden_dim=8):
        self.obs_dim    = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_params = (self.obs_dim * self.hidden_dim + self.hidden_dim + 
                         self.hidden_dim * self.action_dim + self.action_dim)

    def unpack_params(self, genome):
        genome = np.asarray(genome, dtype=np.float64)
        idx = 0
        w1_size = self.obs_dim * self.hidden_dim
        W1 = genome[idx: idx + w1_size].reshape(self.obs_dim, self.hidden_dim)
        idx += w1_size
        b1 = genome[idx: idx + self.hidden_dim]
        idx += self.hidden_dim
        w2_size = self.hidden_dim * self.action_dim
        W2 = genome[idx: idx + w2_size].reshape(self.hidden_dim, self.action_dim)
        idx += w2_size
        b2 = genome[idx: idx + self.action_dim]
        return W1, b1, W2, b2

    def forward(self, obs, genome):
        W1, b1, W2, b2 = self.unpack_params(genome)
        x = np.asarray(obs, dtype=np.float64).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = np.tanh(x @ W1 + b1)
        out = h @ W2 + b2
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).flatten().astype(np.float32)

def run_ga_backtest_episode(env, policy, genome):
    obs, info = env.reset()
    done = False
    fechas, net_worths, rewards_list = [], [], []
    cash_weights, turnovers, transaction_costs = [], [], []
    weights_history, actions_history, ibex_log_rets = [], [], []

    while not done:
        action = policy.forward(obs, genome)
        obs, reward, terminated, truncated, info = env.step(action)

        fechas.append(pd.to_datetime(info["date"]))
        net_worths.append(float(info["net_worth"]))
        rewards_list.append(float(reward))
        cash_weights.append(float(info.get("cash_weight", 0.0)))
        turnovers.append(float(info.get("turnover", 0.0)))
        transaction_costs.append(float(info.get("transaction_cost", 0.0)))
        weights_history.append(np.asarray(info.get("weights", []), dtype=float))
        actions_history.append(np.asarray(action, dtype=float))
        ibex_log_rets.append(float(info.get("ibex_log_ret", 0.0)))
        done = bool(terminated or truncated)

    df_path = pd.DataFrame({
        "Date": fechas, "GA Agent": net_worths, "Reward": rewards_list,
        "CashWeight": cash_weights, "Turnover": turnovers,
        "TransactionCost": transaction_costs, "Index_LogRet": ibex_log_rets,
    }).drop_duplicates("Date").sort_values("Date")
    return df_path, weights_history

def infer_trades_from_weights(df_weights, min_delta=0.02):
    if df_weights.empty or len(df_weights) < 2: return pd.DataFrame()
    asset_cols = [c for c in df_weights.columns if c not in ["Date", "CASH"]]
    trades, trade_id = [], 1
    prev = df_weights.iloc[0]
    for i in range(1, len(df_weights)):
        curr = df_weights.iloc[i]
        dt = curr["Date"]
        for asset in asset_cols:
            old_w = float(prev.get(asset, 0.0)) if pd.notnull(prev.get(asset, np.nan)) else 0.0
            new_w = float(curr.get(asset, 0.0)) if pd.notnull(curr.get(asset, np.nan)) else 0.0
            delta = new_w - old_w
            if abs(delta) >= min_delta:
                trades.append({
                    "Trade_ID": trade_id, "Date": dt, "Ticker": asset,
                    "Trade_Type": "BUY" if delta > 0 else "SELL",
                    "Old_Weight": old_w, "New_Weight": new_w,
                    "Delta_Weight": delta, "Abs_Delta_Weight": abs(delta),
                })
                trade_id += 1
        prev = curr.copy()
    return pd.DataFrame(trades)

def calc_corrected_metrics(date_series, capital_series, rf_annual=0.02):
    s = pd.DataFrame({"Date": pd.to_datetime(date_series), "Capital": capital_series}).dropna()
    s = s.sort_values("Date").drop_duplicates("Date")
    s["returns"] = s["Capital"].pct_change().fillna(0.0)

    dias_naturales = (s["Date"].iloc[-1] - s["Date"].iloc[0]).days
    years = dias_naturales / 365.25 if dias_naturales > 0 else np.nan

    total_return = ((s["Capital"].iloc[-1] / s["Capital"].iloc[0]) - 1) * 100
    annualized_return = (((s["Capital"].iloc[-1] / s["Capital"].iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else np.nan)
    volatility = s["returns"].std(ddof=1) * np.sqrt(252) * 100
    excess_returns = s["returns"] - ((1 + rf_annual) ** (1 / 365.25) - 1)
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / (s["returns"].std(ddof=1) + 1e-12)
    max_drawdown = ((1 + s["returns"]).cumprod() / (1 + s["returns"]).cumprod().cummax() - 1).min() * 100
    win_rate = (s["returns"] > 0).sum() / len(s["returns"]) * 100

    return {
        "Rentabilidad total (%)": total_return, "Rentabilidad Anualizada (%)": annualized_return,
        "Volatilidad (%)": volatility, "Ratio Sharpe": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown, "Tasa de Acierto (%)": win_rate, "Valor Final (EUR)": s["Capital"].iloc[-1],
    }

# ============================================================
# 3. EJECUCIÓN DEL BACKTEST (ZERO-SHOT)
# ============================================================
print("="*70)
print(f"INICIANDO SIMULACIÓN ZERO-SHOT EN: {NOMBRE_INDICE_TEST}")
print("="*70)

# Cargar configuración y genoma
with open(RUTA_CONFIG, 'r') as f: config = json.load(f)
best_genome = np.load(RUTA_GENOMA)

# Preparar datos
df_sp = pd.read_csv(RUTA_DATASET)
df_sp['Date'] = pd.to_datetime(df_sp['Date'])
df_sp_bt = df_sp[(df_sp['Date'].dt.year >= config['ANIO_INICIO_BACKTEST']) & 
                 (df_sp['Date'].dt.year <= config['ANIO_FIN_BACKTEST'])].copy()

tickers_sp = sorted(df_sp_bt['Ticker'].unique().tolist())

df_indices = pd.read_csv(RUTA_INDICES)
df_indices['Date'] = pd.to_datetime(df_indices['Date'])
# TRUCO: Forzar el nombre a 'IBEX 35' temporalmente para que el entorno original no se rompa
df_indices['Ticker'] = "IBEX 35" 

# Crear Entorno y Policy
print("-> Configurando Agente y Entorno...")
env = MultiStockTradingEnvGA2(
    df_train=df_sp_bt,
    tickers=tickers_sp,
    df_indices=df_indices,
    features=config['FEATURES_DETECTADAS'],
    initial_balance=config['CAPITAL_INICIAL'],
    commission=config['COMISION'],
    window_size=config['window_size'],
    rebalance_freq=config['rebalance_freq'],
    obs_clip=config['obs_clip'],
    max_positions=config['max_positions'],
    portfolio_init_path=str(RUTA_PORTFOLIO)
)

policy = GeneticMLPPolicy(obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], hidden_dim=config['hidden_dim'])

# Ejecutar Simulación Diaria
print(f"-> Simulando operaciones del Agente en el {NOMBRE_INDICE_TEST}...")
df_ga, weights_history = run_ga_backtest_episode(env, policy, best_genome)
fechas_bt = df_ga["Date"].values

# Extraer Weights
weights_cols = tickers_sp + ["CASH"]
df_weights = pd.DataFrame(weights_history, columns=weights_cols)
df_weights.insert(0, "Date", fechas_bt)

# Extraer Trades
df_trades = infer_trades_from_weights(df_weights, min_delta=0.02)

# ============================================================
# 4. PREPARACIÓN DE MÉTRICAS Y ARCHIVOS DE SALIDA
# ============================================================
print("-> Calculando Benchmarks y Métricas...")
df_prices_bt = df_sp_bt.pivot(index="Date", columns="Ticker", values=config['TARGET_PRICE_COL']).sort_index().reindex(fechas_bt).ffill().dropna(axis=1, how="any")
shares_bought = (config['CAPITAL_INICIAL'] / df_prices_bt.shape[1]) / df_prices_bt.iloc[0].values
bh_series = (df_prices_bt.values * shares_bought).sum(axis=1)

idx_series = df_indices.pivot(index="Date", columns="Ticker", values="Close").reindex(fechas_bt).ffill()
ret = idx_series["IBEX 35"].pct_change().fillna(0.0) # Lo leemos como IBEX 35 por el truco anterior
nw_idx = config['CAPITAL_INICIAL'] * (1 + ret).cumprod()
nw_idx.iloc[0] = config['CAPITAL_INICIAL']

df_comparativo = pd.DataFrame({"Date": fechas_bt})
df_comparativo["Agente AG"] = df_ga["GA Agent"].values
df_comparativo["Buy & Hold"] = bh_series
df_comparativo[NOMBRE_INDICE_TEST] = nw_idx.values # Lo guardamos con su nombre real

df_norm = df_comparativo.copy()
for c in [x for x in df_norm.columns if x != "Date"]:
    df_norm[c] = (df_norm[c] / df_norm[c].iloc[0]) * 100

resultados_metricas = {c: calc_corrected_metrics(df_comparativo["Date"], df_comparativo[c]) for c in df_comparativo.columns if c != "Date"}
df_metricas = pd.DataFrame(resultados_metricas).T

# EXPORTAR LOS 6 EXCEL/CSV
print("-> Exportando los 6 archivos CSV requeridos...")
df_trades.to_csv(BACKTEST_DIR / "all_trades.csv", index=False)
df_ga.to_csv(BACKTEST_DIR / "ga_path_details.csv", index=False)
df_metricas.to_csv(BACKTEST_DIR / "performance_metrics.csv")
df_weights.to_csv(BACKTEST_DIR / "portfolio_composition_history.csv", index=False)
df_norm.to_csv(BACKTEST_DIR / "portfolio_evolution_base100.csv", index=False)
df_comparativo.to_csv(BACKTEST_DIR / "portfolio_evolution.csv", index=False)

# ============================================================
# 5. GENERACIÓN DE GRÁFICOS (S&P 500)
# ============================================================
print("-> Generando visualizaciones...")

# 1. Turnover Cash
plt.figure(figsize=(16, 5))
plt.plot(df_ga["Date"], df_ga["Turnover"] * 100, label="Turnover (%)", color="black")
plt.fill_between(df_ga["Date"], 0, df_ga["CashWeight"] * 100, alpha=0.25, color="red", label="Cash Weight (%)")
plt.title(f"Turnover y Liquidez (CASH) — {NOMBRE_INDICE_TEST}", fontsize=13, fontweight="bold")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(BACKTEST_DIR / "turnover_cash.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. Portfolio Composition Evolution
df_plot = df_weights.set_index("Date")
activos_operados = [c for c in df_plot.columns if c != "CASH" and df_plot[c].max() > 0.001]
df_plot_active = df_plot[activos_operados]
colores_stack = (plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors)

plt.figure(figsize=(18, 8))
plt.stackplot(df_plot_active.index, df_plot_active.T.values, labels=df_plot_active.columns, colors=colores_stack[:len(activos_operados)], alpha=0.85)
plt.legend(loc="upper left", ncol=max(4, min(10, len(activos_operados) // 2)), fontsize=8)
plt.title(f"Evolución de Composición ({len(activos_operados)} activos operados) — {NOMBRE_INDICE_TEST}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(BACKTEST_DIR / "portfolio_composition_evolution.png", dpi=300, bbox_inches="tight")
plt.close()

# 3. Composición Cartera en Barras
last_row = df_plot.iloc[-1]
last_row = last_row[last_row > 0.01].sort_values(ascending=True)
if not last_row.empty:
    plt.figure(figsize=(10, 6))
    bars = plt.barh(last_row.index, last_row.values * 100, color="steelblue", edgecolor="black", alpha=0.85)
    
    # --- NUEVO: Cambiar tamaño de los nombres de los activos (Eje Y) ---
    plt.yticks(fontsize=16) 
    
    for bar in bars: 
        # --- NUEVO: Añadido fontsize=11 para el texto de los porcentajes ---
        plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, 
                 f"{bar.get_width():.2f}%", va="center", fontsize=11)
        
    plt.title(f"Composición Final de la Cartera — {NOMBRE_INDICE_TEST}", fontsize=20, fontweight="bold")
    plt.tight_layout()
    plt.savefig(BACKTEST_DIR / "composicion_final_barras.png", dpi=200, bbox_inches="tight")
    plt.close()
    
# 4. Correlación Cartera Final
active_tickers = last_row[last_row.index != 'CASH'].index.tolist()
if len(active_tickers) >= 2:
    df_pivot = df_prices_bt[active_tickers]
    corr_matrix = df_pivot.pct_change().dropna().corr()
    
    fig, ax = plt.subplots(figsize=(max(8, int(len(active_tickers)*0.6)), max(7, int(len(active_tickers)*0.5))))
    
    # --- NUEVO: annot_kws={"size": 11} para los números de dentro de las cajas ---
    sns.heatmap(corr_matrix, mask=np.triu(np.ones_like(corr_matrix, dtype=bool), k=1), 
                annot=(len(active_tickers)<=15), annot_kws={"size": 11}, fmt='.2f', 
                cmap='RdBu', center=0, vmin=-1, vmax=1, square=True, ax=ax)
    
    # --- NUEVO: Cambiar tamaño de los nombres de los activos (Ejes X e Y) ---
    ax.tick_params(axis='both', labelsize=16) 
    
    # Truco extra: rotar las etiquetas del eje X para que no se pisen si son muchas
    plt.xticks(rotation=45, ha='right')
    
    ax.set_title(f'Matriz de Correlación Cartera Final — {NOMBRE_INDICE_TEST}', fontsize=20, fontweight='bold', pad=14)
    plt.tight_layout()
    plt.savefig(BACKTEST_DIR / 'correlacion_cartera_final.png', dpi=300, bbox_inches='tight')
    plt.close()
# 5. Evolución Base 100 
plt.figure(figsize=(16, 7))
for c in [x for x in df_norm.columns if x != "Date"]:
    plt.plot(df_norm["Date"], df_norm[c], label=c, linewidth=(2.5 if c == "Agente AG" else 1.5), linestyle=("-" if c == "Agente AG" else "--"))
    plt.axhline(100, color="black", linestyle=":", alpha=0.6)
plt.title(f"Evolución Normalizada (Base 100) — Agente vs {NOMBRE_INDICE_TEST}", fontsize=14, fontweight="bold")
plt.legend(fontsize=10, ncol=2); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(BACKTEST_DIR / "grafico_base100.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 6. COMPARATIVA: ENTRENAMIENTO (IBEX) vs TEST (S&P 500)
# ============================================================
print("-> Generando comparativa transversal...")
if RUTA_BACKTEST_ORIGINAL.exists():
    df_origen = pd.read_csv(RUTA_BACKTEST_ORIGINAL)
    df_origen['Date'] = pd.to_datetime(df_origen['Date'])
    
    df_cruce = pd.DataFrame({"Date": df_norm["Date"]})
    df_cruce[NOMBRE_TEST] = df_norm["Agente AG"]
    
    col_agente = "Agente AG" if "Agente AG" in df_origen.columns else ("GA Agent" if "GA Agent" in df_origen.columns else None)
    if col_agente:
        df_origen_agente = df_origen[['Date', col_agente]].rename(columns={col_agente: NOMBRE_ENTRENAMIENTO})
        df_cruce = pd.merge(df_cruce, df_origen_agente, on="Date", how="left").ffill()
        
        df_cruce.to_csv(DIR_COMPARATIVA / "datos_comparativa_modelos.csv", index=False)
        
        plt.figure(figsize=(16, 7))
        plt.plot(df_cruce["Date"], df_cruce[NOMBRE_ENTRENAMIENTO], label=f"{NOMBRE_ENTRENAMIENTO} (Datos de Origen)", linewidth=2, color="purple")
        plt.plot(df_cruce["Date"], df_cruce[NOMBRE_TEST], label=f"{NOMBRE_TEST} (Zero-Shot)", linewidth=2.5, color="steelblue")
        plt.axhline(100, color="black", linestyle=":", alpha=0.6)
        plt.title(f"Comparativa de Generalización: {NOMBRE_ENTRENAMIENTO} vs {NOMBRE_TEST}", fontsize=14, fontweight="bold")
        plt.ylabel("Evolución (Base 100)")
        plt.legend(fontsize=11); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(DIR_COMPARATIVA / "grafico_comparativo_modelos.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✔ Comparativa estructural guardada en: {DIR_COMPARATIVA}")
else:
    print(f"⚠️ No se encontró el backtest original en {RUTA_BACKTEST_ORIGINAL}")

print("="*70)
print(f"✅ PROCESO FINALIZADO. Todos los CSVs y gráficos están en: {BACKTEST_DIR}")