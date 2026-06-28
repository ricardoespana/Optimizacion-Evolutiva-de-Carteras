# -*- coding: utf-8 -*-
"""
train_ga_multistock_v2.py

Algoritmo Genético multi-activo para batir al IBEX 35.
Usa MultiStockTradingEnvGA2 como entorno.

Diferencias respecto a ga_estable.py:
- Entorno: MultiStockTradingEnvGA2 (reward = diferencial log-ret cartera - IBEX 35).
- Cartera inicial desde portfolio/portfolio.inicial.csv (columna Peso_Inicial).
- Sin límite de cardinalidad por defecto (max_positions=None).
- Split temporal: Train 2021-2023, Backtest 2024-2025.
- Función de fitness orientada a maximizar el alpha acumulado vs IBEX 35.
"""

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import sys, os

from trading_env_multistock_ga_v3 import MultiStockTradingEnvGA2

_t_inicio_total = time.perf_counter()
# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

ID_EXPERIMENTO = "GA_MultiStock_v31" #MODIFICAR POR CADA EJECUCIÓN

DIR_BASE         = Path("GA_Env")
DIR_DATOS        = DIR_BASE / "data"
DIR_EXPERIMENTOS = DIR_BASE / "experimentos"
DIR_SALIDA       = DIR_EXPERIMENTOS / ID_EXPERIMENTO
DIR_SALIDA.mkdir(parents=True, exist_ok=True)

RUTA_DATASET      = DIR_DATOS / "dataset_GA_estandarizado_v4.csv"
RUTA_INDICES      = DIR_DATOS / "indices_cierre.csv"
RUTA_RESULTADOS   = DIR_SALIDA / "ga_results.csv"
RUTA_MEJOR_GENOMA = DIR_SALIDA / "best_genome.npy"
RUTA_MEJOR_CONFIG = DIR_SALIDA / "best_config_ga.json"
RUTA_DATASET_INFO = DIR_SALIDA / "dataset_schema_detected_ga.json"
RUTA_ERRORES      = DIR_SALIDA / "errores_evaluacion.csv"

DEFAULT_PARAMS = {
    # ── Períodos temporales ───────────────────────────────────────────────────
    "ANIO_INICIO_TRAIN"  : 2021,
    "ANIO_FIN_TRAIN"     : 2023,
    "ANIO_INICIO_BACKTEST": 2024,
    "ANIO_FIN_BACKTEST"  : 2025,

    # ── Capital y costes ─────────────────────────────────────────────────────
    "CAPITAL_INICIAL"    : 100_000.0,
    "COMISION"           : 0.001,
    "SEED"               : 123,

    # ── Filtros de tickers ────────────────────────────────────────────────────
    "MIN_TICKER_OBS_TRAIN"   : 120,
    "MIN_TICKER_OBS_BACKTEST": 60,

    # ── Columnas del dataset ──────────────────────────────────────────────────
    "USE_CLOSE_AS_FEATURE"               : False,
    "USE_OPEN_HIGH_LOW_VOLUME_AS_FEATURES": False,
    "NON_NUMERIC_KNOWN_COLS"             : ["Date", "Ticker", "RIC", "Nombre",
                                            "Sector", "Industria", "Indice"],
    "TARGET_PRICE_COL"                   : "Close",

    # ── Parámetros del entorno ────────────────────────────────────────────────
    "hidden_dim"    : 8,
    "window_size"   : 10, #CUANTOS DÍAS PUEDE VER HACIA ATRAS
    "rebalance_freq": 20, # =5 FRECUENCIA SEMANAL  =20 FRECUENCIA MENSUAL
    "obs_clip"      : 5.0,
    "max_positions" : 10,    # None = sin límite de cardinalidad

    # ── Parámetros del algoritmo genético ────────────────────────────────────
    "ga_population"         : 100,
    "ga_generations"        : 500,
    "ga_elite_size"         : 2,
    "ga_tournament_size"    : 3,
    "ga_crossover_alpha"    : 0.40,
    "ga_mutation_prob"      : 0.025  ,#Probabilidad de mutación, incrementar si convergencia prematura
    "ga_mutation_std"       : 0.05,
    "eval_seeds"            : 1,
    "invalid_fitness_penalty": -1e9,
    "genome_init_scale"     : 0.03,
}
# ============================================================
# PARAMETROS DE LA FUNCIÓN FITNESS
# ============================================================


# ============================================================
# UTILIDADES
# ============================================================

def to_python_types(obj):
    if isinstance(obj, dict):
        return {str(k): to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_python_types(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj

def save_json_safe(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_python_types(obj), f, indent=4, ensure_ascii=False)

def set_seed(seed):
    np.random.seed(int(seed))

# ============================================================
# DATOS
# ============================================================

def load_dataset():
    if not RUTA_DATASET.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {RUTA_DATASET}")

    df = pd.read_csv(RUTA_DATASET)

    required = {"Date", "Ticker", DEFAULT_PARAMS["TARGET_PRICE_COL"]}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("La columna Date contiene valores no parseables.")

    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

def load_indices():
    """Carga el CSV de índices eliminando la fila de cabecera duplicada si existe."""
    if not RUTA_INDICES.exists():
        raise FileNotFoundError(f"No se encontró el CSV de índices en {RUTA_INDICES}")

    df_idx = pd.read_csv(RUTA_INDICES)
    df_idx.columns = [str(c).strip() for c in df_idx.columns]
    df_idx["Date"]  = pd.to_datetime(df_idx["Date"], errors="coerce")
    df_idx["Close"] = pd.to_numeric(df_idx["Close"], errors="coerce")
    df_idx = df_idx.dropna(subset=["Date", "Close"])
    return df_idx

def detect_dataset_schema(df, use_close_as_feature=False, use_ohlcv_as_features=False):
    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    excluded  = ["Date", "Ticker"]
    excluded += [c for c in DEFAULT_PARAMS["NON_NUMERIC_KNOWN_COLS"] if c in df.columns]

    if not use_ohlcv_as_features:
        excluded += [c for c in ["Open", "High", "Low", "Volume"] if c in df.columns]

    if not use_close_as_feature and DEFAULT_PARAMS["TARGET_PRICE_COL"] in df.columns:
        excluded += [DEFAULT_PARAMS["TARGET_PRICE_COL"]]

    feature_candidates = [c for c in numeric_cols if c not in excluded]
    constant_cols      = [c for c in feature_candidates if df[c].nunique(dropna=True) <= 1]
    feature_candidates = [c for c in feature_candidates if c not in constant_cols]

    return {
        "all_columns"                         : df.columns.tolist(),
        "numeric_columns"                     : numeric_cols,
        "non_numeric_columns"                 : non_numeric_cols,
        "excluded_from_features"              : sorted(list(set(excluded))),
        "feature_candidates"                  : feature_candidates,
        "constant_feature_candidates_removed" : constant_cols,
        "n_rows"                              : int(len(df)),
        "n_tickers"                           : int(df["Ticker"].nunique()),
        "date_min"                            : str(df["Date"].min()),
        "date_max"                            : str(df["Date"].max()),
    }

def clean_numeric_columns(df, columns):
    out = df.copy()
    for c in columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    return out

def validate_panel(df, feature_cols):
    dup = int(df.duplicated(subset=["Date", "Ticker"]).sum())
    if dup > 0:
        raise ValueError(f"Hay {dup} duplicados Date-Ticker en el dataset.")
    check_cols = [DEFAULT_PARAMS["TARGET_PRICE_COL"]] + feature_cols
    nulls      = df[check_cols].isna().mean().sort_values(ascending=False)
    return nulls

def filter_by_years(df, start_year, end_year):
    return df[(df["Date"].dt.year >= start_year) & (df["Date"].dt.year <= end_year)].copy()

def eligible_tickers(df_train, df_backtest, min_obs_train=120, min_obs_backtest=60):
    train_counts    = df_train.groupby("Ticker").size()
    backtest_counts = df_backtest.groupby("Ticker").size()
    common          = sorted(set(train_counts.index).intersection(backtest_counts.index))
    return [
        t for t in common
        if train_counts[t] >= min_obs_train and backtest_counts[t] >= min_obs_backtest
    ]

def restrict_to_tickers(df, tickers):
    return df[df["Ticker"].isin(tickers)].copy()

# ============================================================
# ENTORNO
# ============================================================

def make_env(df, tickers, params, feature_cols, df_indices, seed=None):
    """
    Instancia MultiStockTradingEnvGA2.
    Todos los parámetros se toman de DEFAULT_PARAMS para máxima parametrización.
    """
    env = MultiStockTradingEnvGA2(
        df_train           = df,
        tickers            = tickers,
        df_indices         = df_indices,
        features           = feature_cols,
        initial_balance    = float(params["CAPITAL_INICIAL"]),
        commission         = float(params["COMISION"]),
        risk_free_rate     = 0.02,
        window_size        = int(params["window_size"]),
        rebalance_freq     = int(params["rebalance_freq"]),
        obs_clip           = float(params["obs_clip"]),
        max_positions      = params["max_positions"],
        #portfolio_init_path= str(RUTA_PORTFOLIO), NO SE USA
    )

    if seed is not None:
        env.reset(seed=int(seed))

    return env

# ============================================================
# BENCHMARKS Y MÉTRICAS
# ============================================================

def calculate_equal_weight(df_eval, fechas_bt, capital_inicial):
    df_sorted  = df_eval.sort_values(["Ticker", "Date"]).copy()
    df_sorted["Retorno_Diario"] = (
        df_sorted.groupby("Ticker")[DEFAULT_PARAMS["TARGET_PRICE_COL"]].pct_change()
    )
    df_bench    = df_sorted[df_sorted["Date"].isin(pd.to_datetime(fechas_bt))].copy()
    retornos_ew = (
        df_bench.groupby("Date")["Retorno_Diario"]
        .mean()
        .reindex(pd.to_datetime(fechas_bt))
        .fillna(0.0)
    )
    ew_net_worth = capital_inicial * (1 + retornos_ew).cumprod()
    if len(ew_net_worth) > 0:
        ew_net_worth.iloc[0] = capital_inicial
    return ew_net_worth.values

def calculate_buyhold(df_eval, fechas_bt, capital_inicial):
    df_prices = (
        df_eval.pivot(index="Date", columns="Ticker",
                      values=DEFAULT_PARAMS["TARGET_PRICE_COL"])
        .sort_index()
    )
    df_prices_aligned = (
        df_prices.reindex(df_prices.index.union(pd.to_datetime(fechas_bt)))
        .sort_index()
        .ffill()
        .reindex(pd.to_datetime(fechas_bt))
    )
    df_prices_aligned = df_prices_aligned.dropna(axis=1, how="any")

    if df_prices_aligned.empty:
        return np.full(len(fechas_bt), capital_inicial, dtype=float)

    n_assets       = df_prices_aligned.shape[1]
    initial_prices = df_prices_aligned.iloc[0].values
    shares_bought  = (capital_inicial / n_assets) / initial_prices
    return (df_prices_aligned.values * shares_bought).sum(axis=1)

def calculate_index_series(df_indices, fechas_bt, capital_inicial):
    df_idx = df_indices.copy()
    df_idx = df_idx.dropna(subset=["Date"]).drop_duplicates(
        subset=["Date", "Ticker"], keep="last"
    ).sort_values(["Ticker", "Date"])

    df_idx_pivot   = df_idx.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    df_idx_aligned = (
        df_idx_pivot.reindex(df_idx_pivot.index.union(pd.to_datetime(fechas_bt)))
        .sort_index()
        .ffill()
        .reindex(pd.to_datetime(fechas_bt))
    )
    df_idx_aligned = df_idx_aligned.dropna(axis=1, how="any")

    out = {}
    for col in df_idx_aligned.columns:
        ret = df_idx_aligned[col].pct_change().fillna(0.0)
        nw  = capital_inicial * (1 + ret).cumprod()
        if len(nw) > 0:
            nw.iloc[0] = capital_inicial
        out[col] = nw.values
    return out

def compute_metrics(date_series, capital_series, rf_annual=0.02):
    s = pd.DataFrame({
        "Date"   : pd.to_datetime(date_series),
        "Capital": np.asarray(capital_series, dtype=float),
    }).dropna()
    s = s.sort_values("Date").drop_duplicates("Date")

    if len(s) < 2 or not np.all(np.isfinite(s["Capital"].values)):
        return {
            "total_return": -999.0, "cagr": -999.0,
            "volatility"  : 999.0,  "sharpe": -999.0, "max_drawdown": -1.0,
        }

    s["ret"]      = s["Capital"].pct_change().fillna(0.0)
    total_return  = s["Capital"].iloc[-1] / max(s["Capital"].iloc[0], 1e-12) - 1.0
    n_days        = (s["Date"].iloc[-1] - s["Date"].iloc[0]).days
    cagr          = (
        (s["Capital"].iloc[-1] / max(s["Capital"].iloc[0], 1e-12)) ** (365.25 / n_days) - 1
        if n_days > 0 else -999.0
    )
    vol           = s["ret"].std(ddof=0) * np.sqrt(252)
    rf_daily      = (1 + rf_annual) ** (1 / 252) - 1
    ret_std       = s["ret"].std(ddof=0)
    excess        = s["ret"] - rf_daily
    sharpe        = (excess.mean() / ret_std) * np.sqrt(252) if ret_std > 1e-12 else -999.0
    running_max   = s["Capital"].cummax()
    max_dd        = (s["Capital"] / running_max - 1).min()

    return {
        "total_return": float(total_return),
        "cagr"        : float(cagr),
        "volatility"  : float(vol),
        "sharpe"      : float(sharpe),
        "max_drawdown": float(max_dd),
    }

# ============================================================
# POLICY GENÉTICA (MLP de 2 capas)
# ============================================================

class GeneticMLPPolicy:
    def __init__(self, obs_dim, action_dim, hidden_dim=8):
        self.obs_dim    = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        self.n_params = (
            self.obs_dim * self.hidden_dim
            + self.hidden_dim
            + self.hidden_dim * self.action_dim
            + self.action_dim
        )

    def unpack_params(self, genome):
        genome = np.asarray(genome, dtype=np.float64)
        if genome.size != self.n_params:
            raise ValueError(f"Genome size {genome.size} != expected {self.n_params}")

        idx     = 0
        w1_size = self.obs_dim * self.hidden_dim
        W1      = genome[idx: idx + w1_size].reshape(self.obs_dim, self.hidden_dim)
        idx    += w1_size
        b1      = genome[idx: idx + self.hidden_dim]
        idx    += self.hidden_dim
        w2_size = self.hidden_dim * self.action_dim
        W2      = genome[idx: idx + w2_size].reshape(self.hidden_dim, self.action_dim)
        idx    += w2_size
        b2      = genome[idx: idx + self.action_dim]

        return W1, b1, W2, b2

    def forward(self, obs, genome):
        W1, b1, W2, b2 = self.unpack_params(genome)
        x = np.asarray(obs, dtype=np.float64).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = np.tanh(x @ W1 + b1)
        out = h @ W2 + b2
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.flatten().astype(np.float32)

    def random_genome(self, rng, scale=0.03):
        return rng.normal(0.0, scale, size=self.n_params).astype(np.float64)

# ============================================================
# ALGORITMO GENÉTICO (codificación real)
# ============================================================

class RealCodedGA:
    def __init__(
        self,
        n_params,
        pop_size=20,
        elite_size=4,
        tournament_size=3,
        crossover_alpha=0.30,
        mutation_prob=0.03,
        mutation_std=0.01,
        seed=42,
    ):
        self.n_params       = int(n_params)
        self.pop_size       = int(pop_size)
        self.elite_size     = int(elite_size)
        self.tournament_size = int(tournament_size)
        self.crossover_alpha = float(crossover_alpha)
        self.mutation_prob  = float(mutation_prob)
        self.mutation_std   = float(mutation_std)
        self.rng            = np.random.default_rng(seed)

        if self.elite_size >= self.pop_size:
            raise ValueError("elite_size debe ser menor que pop_size")
        if self.tournament_size > self.pop_size:
            raise ValueError("tournament_size no puede ser mayor que pop_size")

    def initialize(self, genome_generator, init_scale=0.03):
        return [genome_generator(self.rng, scale=init_scale) for _ in range(self.pop_size)]

    def tournament_select(self, population, fitnesses):
        idxs           = self.rng.choice(len(population), size=self.tournament_size, replace=False)
        best_local_idx = idxs[np.argmax([fitnesses[i] for i in idxs])]
        return population[best_local_idx].copy()

    def crossover(self, parent1, parent2):
        alpha  = self.crossover_alpha
        gamma  = self.rng.uniform(-alpha, 1 + alpha, size=parent1.shape)
        child1 = gamma * parent1 + (1.0 - gamma) * parent2
        child2 = gamma * parent2 + (1.0 - gamma) * parent1
        return child1, child2

    def mutate(self, genome):
        mask  = self.rng.random(genome.shape) < self.mutation_prob
        noise = self.rng.normal(0.0, self.mutation_std, size=genome.shape)
        out   = genome.copy()
        out[mask] += noise[mask]
        return out

    def next_generation(self, population, fitnesses):
        order      = np.argsort(fitnesses)[::-1]
        population = [population[i] for i in order]
        fitnesses  = [fitnesses[i] for i in order]

        new_pop = [population[i].copy() for i in range(self.elite_size)]

        while len(new_pop) < self.pop_size:
            p1     = self.tournament_select(population, fitnesses)
            p2     = self.tournament_select(population, fitnesses)
            c1, c2 = self.crossover(p1, p2)
            c1     = self.mutate(c1)
            c2     = self.mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < self.pop_size:
                new_pop.append(c2)

        return new_pop[:self.pop_size], population[0].copy(), float(fitnesses[0])

# ============================================================
# EVALUACIÓN DE GENOMAS
# ============================================================

def run_policy_episode(env, policy, genome, max_steps=None):
    obs, info = env.reset()
    done      = False

    dates, net_worths, turnovers, cash_weights, rewards_list = [], [], [], [], []
    ibex_log_rets = []
    steps = 0

    while not done:
        action = policy.forward(obs, genome)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

        obs, reward, terminated, truncated, info = env.step(action)

        dates.append(pd.to_datetime(info["date"]))
        net_worths.append(float(info["net_worth"]))
        turnovers.append(float(info.get("turnover", 0.0)))
        cash_weights.append(float(info.get("cash_weight", 0.0)))
        rewards_list.append(float(reward))
        ibex_log_rets.append(float(info.get("ibex_log_ret", 0.0)))

        done = bool(terminated or truncated)
        steps += 1

        if max_steps is not None and steps >= max_steps:
            break

    return {
        "dates"        : np.array(dates),
        "net_worths"   : np.array(net_worths, dtype=float),
        "turnovers"    : np.array(turnovers, dtype=float),
        "cash_weights" : np.array(cash_weights, dtype=float),
        "rewards"      : np.array(rewards_list, dtype=float),
        "ibex_log_rets": np.array(ibex_log_rets, dtype=float),
    }

def evaluate_genome(genome, policy, env, benchmark_cagrs, params, seeds=None):
    """
    Versión ultrarrápida: usa un entorno pre-instanciado y benchmarks precalculados.
    """
    penalty = float(params.get("invalid_fitness_penalty", -1e9))
    if seeds is None:
        seeds = [int(params["SEED"])]

    details_list = []
    error_rows   = []

    for seed in seeds:
        try:
            # 1. RESET DEL ENTORNO (Operación ultrarrápida de microsegundos)
            env.reset(seed=seed)
            rollout = run_policy_episode(env, policy, genome)

            dates      = rollout["dates"]
            ga_series  = rollout["net_worths"]

            if len(dates) < 2 or len(ga_series) < 2:
                raise ValueError("Rollout demasiado corto para evaluar.")

            # 2. MÉTRICAS DEL AGENTE
            ga_m = compute_metrics(dates, ga_series)

            # 3. USO DE BENCHMARKS PRECALCULADOS (Cero carga de procesamiento)
            ibex35_cagr    = benchmark_cagrs.get("ibex35_cagr", -999.0)
            benchmark_cagr = benchmark_cagrs.get("best_cagr", -999.0)

            alpha_vs_best   = ga_m["cagr"] - benchmark_cagr
            alpha_vs_ibex35 = ga_m["cagr"] - ibex35_cagr if np.isfinite(ibex35_cagr) else alpha_vs_best

            turnover_mean    = float(np.mean(rollout["turnovers"]))    if len(rollout["turnovers"])    else 0.0
            cash_mean        = float(np.mean(rollout["cash_weights"])) if len(rollout["cash_weights"]) else 0.0
            reward_acumulada = float(np.sum(rollout["rewards"]))       

            # 4. FUNCIÓN DE FITNESS
            score = (
                        100.0 * alpha_vs_ibex35
                        + 5.0 * float(ga_m["sharpe"])
                        - 1.0 * abs(ga_m["max_drawdown"])
                        - 0.01 * turnover_mean
                        + 10.0 * reward_acumulada
                    )

            if not np.isfinite(score):
                raise ValueError("El score calculado no es finito.")

            details_list.append({
                "score"                 : float(score),
                "alpha_vs_ibex35"       : float(alpha_vs_ibex35),
                "ga_cagr"               : float(ga_m["cagr"]),
                "ga_sharpe"             : float(ga_m["sharpe"]),
                "ga_mdd"                : float(ga_m["max_drawdown"]),
                "reward_acumulada"      : float(reward_acumulada),
                "turnover_mean"         : float(turnover_mean),
                "ok"                    : True,
            })

        except Exception as e:
            # ... (Mismo manejo de errores que tenías) ...
            details_list.append({
                "score": float(penalty), "ok": False
            })

    valid_scores = [d["score"] for d in details_list if d["ok"]]
    final_score  = float(np.mean(valid_scores)) if valid_scores else float(penalty)

    return {
        "score": final_score,
        "n_valid": int(sum(d["ok"] for d in details_list)),
        "details_by_seed": details_list,
    }
# ============================================================
# PIPELINE DE PREPARACIÓN
# ============================================================

def prepare_data(params):
    df = load_dataset()

    schema = detect_dataset_schema(
        df,
        use_close_as_feature    = bool(params["USE_CLOSE_AS_FEATURE"]),
        use_ohlcv_as_features   = bool(params["USE_OPEN_HIGH_LOW_VOLUME_AS_FEATURES"]),
    )
    save_json_safe(schema, RUTA_DATASET_INFO)

    feature_cols = schema["feature_candidates"]
    if not feature_cols:
        raise ValueError("No se detectaron features numéricas válidas.")

    df       = clean_numeric_columns(df, [params["TARGET_PRICE_COL"]] + feature_cols)
    null_report = validate_panel(df, feature_cols)

    df_train    = filter_by_years(df, params["ANIO_INICIO_TRAIN"],    params["ANIO_FIN_TRAIN"])
    df_backtest = filter_by_years(df, params["ANIO_INICIO_BACKTEST"], params["ANIO_FIN_BACKTEST"])

    if df_train.empty or df_backtest.empty:
        raise ValueError("Alguno de los conjuntos train/backtest está vacío.")

    tickers = eligible_tickers(
        df_train, df_backtest,
        min_obs_train    = int(params["MIN_TICKER_OBS_TRAIN"]),
        min_obs_backtest = int(params["MIN_TICKER_OBS_BACKTEST"]),
    )
    if not tickers:
        raise ValueError("No hay tickers elegibles tras aplicar los filtros mínimos.")

    df_train    = restrict_to_tickers(df_train,    tickers)
    df_backtest = restrict_to_tickers(df_backtest, tickers)

    return df_train, df_backtest, tickers, feature_cols, schema, null_report

# ============================================================
# ENTRENAMIENTO
# ============================================================
 
def train_ga(df_train, df_backtest, tickers, feature_cols, params, df_indices):
    
    """
    Entrena el GA sobre df_train y evalúa el fitness sobre df_backtest.
    Separa los datos de entrenamiento (2021-2023) y backtest (2024-2025)
    tal y como se especifica en los parámetros.
    """
    # 1. CREAR EL ENTORNO UNA SOLA VEZ
    print("Construyendo entorno de entrenamiento (Caché Numpy)...")
    env_train = make_env(df_train, tickers, params, feature_cols, df_indices, seed=params["SEED"])
    obs_dim    = int(env_train.observation_space.shape[0])
    action_dim = int(env_train.action_space.shape[0])
    policy = GeneticMLPPolicy(
        obs_dim    = obs_dim,
        action_dim = action_dim,
        hidden_dim = int(params["hidden_dim"]),
    )

    ga = RealCodedGA(
        n_params        = policy.n_params,
        pop_size        = int(params["ga_population"]),
        elite_size      = int(params["ga_elite_size"]),
        tournament_size = int(params["ga_tournament_size"]),
        crossover_alpha = float(params["ga_crossover_alpha"]),
        mutation_prob   = float(params["ga_mutation_prob"]),
        mutation_std    = float(params["ga_mutation_std"]),
        seed            = int(params["SEED"]),
    )

    population = ga.initialize(policy.random_genome, init_scale=float(params["genome_init_scale"]))
    
    eval_seeds = [int(params["SEED"] + i) for i in range(int(params["eval_seeds"]))]
    
    # 2. PRECALCULAR BENCHMARKS DEL PERIODO DE TRAIN UNA SOLA VEZ
    print("Precalculando benchmarks estáticos...")
    # Aproximamos las fechas de train usando el tensor del entorno
    fechas_train = env_train.fechas[env_train.window_size:] 
    
    ew_series  = calculate_equal_weight(df_train, fechas_train, params["CAPITAL_INICIAL"])
    bh_series  = calculate_buyhold(df_train, fechas_train, params["CAPITAL_INICIAL"])
    idx_dict   = calculate_index_series(df_indices, fechas_train, params["CAPITAL_INICIAL"])
    
    ew_m = compute_metrics(fechas_train, ew_series)
    bh_m = compute_metrics(fechas_train, bh_series)
    
    ibex35_cagr = -999.0
    if "IBEX 35" in idx_dict:
        ibex35_cagr = compute_metrics(fechas_train, idx_dict["IBEX 35"])["cagr"]
        
    benchmark_cagrs = {
        "ibex35_cagr": ibex35_cagr,
        "best_cagr": max(ew_m["cagr"], bh_m["cagr"], ibex35_cagr)
    }

    history             = []
    best_global_genome  = None
    best_global_score   = float(params["invalid_fitness_penalty"])

    
    print("=" * 70)
    print("INICIO DEL ENTRENAMIENTO GA v2 — Objetivo: batir al IBEX 35")
    print("=" * 70)
    print("INICIO DEL ENTRENAMIENTO GA v2 — Objetivo: batir al IBEX 35")
    print("=" * 70)
    print(f"Train  : {params['ANIO_INICIO_TRAIN']}–{params['ANIO_FIN_TRAIN']}")
    print(f"Backtest: {params['ANIO_INICIO_BACKTEST']}–{params['ANIO_FIN_BACKTEST']}")
    print(f"Población: {params['ga_population']} | Generaciones: {params['ga_generations']}")
    print("=" * 70)
    
    for gen in range(int(params["ga_generations"])):
        t0 = time.perf_counter()
        fitnesses = []
        valid_counts = []

        # 3. EVALUACIÓN PASANDO EL ENTORNO YA CREADO
        for genome in population:
            details = evaluate_genome(
                genome          = genome,
                policy          = policy,
                env             = env_train,         # Pasamos el entorno en memoria
                benchmark_cagrs = benchmark_cagrs,   # Pasamos los benchmarks en memoria
                params          = params,
                seeds           = eval_seeds,
            )
            
            score = details["score"]
            if not np.isfinite(score):
                score = float(params["invalid_fitness_penalty"])
            fitnesses.append(float(score))
            valid_counts.append(int(details["n_valid"]))

        best_idx_gen    = int(np.argmax(fitnesses))
        best_score_gen  = float(fitnesses[best_idx_gen])
        best_genome_gen = population[best_idx_gen].copy()

        if best_global_genome is None or best_score_gen > best_global_score:
            best_global_score  = best_score_gen
            best_global_genome = best_genome_gen.copy()

        row = {
            "generation"        : gen + 1,
            "best_score"        : best_score_gen,
            "mean_score"        : float(np.mean(fitnesses)),
            "std_score"         : float(np.std(fitnesses)),
            "valid_individuals" : int(sum(v > 0 for v in valid_counts)),
            "elapsed_sec"       : float(time.perf_counter() - t0),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(RUTA_RESULTADOS, index=False)
        # Escritura del resumen de genomas en .txt
        ruta_resumen_genomas = DIR_SALIDA / "evolucion_genomas.txt"
        with open(ruta_resumen_genomas, "a", encoding="utf-8") as f:
                f.write(
        f"Gen {gen+1:>3}/{params['ga_generations']} | "
        f"best={row['best_score']:+.6f} | "
        f"mean={row['mean_score']:+.6f} | "
        f"std={row['std_score']:.6f} | "
        f"valid={row['valid_individuals']}/{len(population)} | "
        f"t={row['elapsed_sec']:.1f}s\n"
    )
        print(
            f"[GA] Gen {gen+1:>3}/{params['ga_generations']} | "
            f"best={row['best_score']:+.6f} | "
            f"mean={row['mean_score']:+.6f} | "
            f"valid={row['valid_individuals']}/{len(population)} | "
            f"t={row['elapsed_sec']:.1f}s"
        )

        population, _, _ = ga.next_generation(population, fitnesses)

    if best_global_genome is None:
        raise RuntimeError("No se consiguió ningún genoma válido durante el entrenamiento.")

    np.save(RUTA_MEJOR_GENOMA, best_global_genome)

    return {
        "policy"            : policy,
        "best_global_genome": best_global_genome,
        "best_global_score" : float(best_global_score),
        "obs_dim"           : obs_dim,
        "action_dim"        : action_dim,
        "n_policy_params"   : int(policy.n_params),
        "history"           : history,
    }

# ============================================================
# EVALUACIÓN FINAL EN BACKTEST
# ============================================================

def evaluate_final(best_global_genome, policy, df_backtest, tickers,
                   feature_cols, params, df_indices):
    if best_global_genome is None:
        raise RuntimeError("best_global_genome es None; no puede evaluarse en backtest.")

    print("Construyendo entorno final de Backtest para métricas...")
    # 1. Crear el entorno de backtest
    env_backtest = make_env(df_backtest, tickers, params, feature_cols, df_indices, seed=params["SEED"])

    # 2. Precalcular benchmarks del backtest
    fechas_bt = env_backtest.fechas[env_backtest.window_size:]
    ew_series = calculate_equal_weight(df_backtest, fechas_bt, params["CAPITAL_INICIAL"])
    bh_series = calculate_buyhold(df_backtest, fechas_bt, params["CAPITAL_INICIAL"])
    idx_dict  = calculate_index_series(df_indices, fechas_bt, params["CAPITAL_INICIAL"])

    ew_m = compute_metrics(fechas_bt, ew_series)
    bh_m = compute_metrics(fechas_bt, bh_series)

    ibex35_cagr = compute_metrics(fechas_bt, idx_dict["IBEX 35"])["cagr"] if "IBEX 35" in idx_dict else -999.0

    benchmark_cagrs = {
        "ibex35_cagr": ibex35_cagr,
        "best_cagr": max(ew_m["cagr"], bh_m["cagr"], ibex35_cagr)
    }

    # 3. Evaluar usando la función ultrarrápida
    return evaluate_genome(
        genome          = best_global_genome,
        policy          = policy,
        env             = env_backtest,
        benchmark_cagrs = benchmark_cagrs,
        params          = params,
        seeds           = [int(params["SEED"])],
    )
# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(DEFAULT_PARAMS["SEED"])

    print("Cargando índices de referencia...")
    df_indices = load_indices()

    print("Cargando y preparando datos...")
    (df_train, df_backtest,
     tickers_local, feature_cols_local,
     schema, null_report) = prepare_data(DEFAULT_PARAMS)


    training_output = train_ga(
        df_train     = df_train,
        df_backtest  = df_backtest,
        tickers      = tickers_local,
        feature_cols = feature_cols_local,
        params       = DEFAULT_PARAMS,
        df_indices   = df_indices,
    )

    backtest_details = evaluate_final(
        best_global_genome = training_output["best_global_genome"],
        policy             = training_output["policy"],
        df_backtest        = df_backtest,
        tickers            = tickers_local,
        feature_cols       = feature_cols_local,
        params             = DEFAULT_PARAMS,
        df_indices         = df_indices,
    )

    final_config = {
        **DEFAULT_PARAMS,
        "FEATURES_DETECTADAS" : feature_cols_local,
        "TICKERS_ELEGIBLES"   : tickers_local,
        "OBS_DIM"             : training_output["obs_dim"],
        "ACTION_DIM"          : training_output["action_dim"],
        "N_POLICY_PARAMS"     : training_output["n_policy_params"],
        "BEST_TRAIN_SCORE"    : training_output["best_global_score"],
        "BACKTEST_METRICS"    : backtest_details,
        "OUTPUT_DIR"          : str(DIR_SALIDA),
    }
    save_json_safe(final_config, RUTA_MEJOR_CONFIG)

    print("\n" + "=" * 70)
    print("ENTRENAMIENTO Y EVALUACIÓN BACKTEST COMPLETADOS")
    print("=" * 70)
    print(f"Mejor score train   : {training_output['best_global_score']:.6f}")
    print(f"Score backtest      : {backtest_details['score']:.6f}")
    if backtest_details["n_valid"] > 0:
        d = backtest_details["details_by_seed"][0]
        print(f"Alpha vs IBEX 35    : {d.get('alpha_vs_ibex35', float('nan')):.4f}")
        print(f"CAGR cartera        : {d.get('ga_cagr', float('nan')):.4f}")
        print(f"CAGR IBEX 35        : {d.get('ibex35_cagr', float('nan')):.4f}")
        print(f"Sharpe              : {d.get('ga_sharpe', float('nan')):.4f}")
        print(f"Max Drawdown        : {d.get('ga_mdd', float('nan')):.4f}")
        print(f"Reward acumulada    : {d.get('reward_acumulada', float('nan')):.6f}")
    print(f"Resultados en       : {DIR_SALIDA}")
    print(f"Best genome         : {RUTA_MEJOR_GENOMA}")
    print(f"Config final        : {RUTA_MEJOR_CONFIG}")

    # Variables globales para la celda de backtest
    globals()["df_backtest"]         = df_backtest
    globals()["tickers"]             = tickers_local
    globals()["feature_cols"]        = feature_cols_local
    globals()["policy"]              = training_output["policy"]
    globals()["best_global_genome"]  = training_output["best_global_genome"]
    globals()["df_indices"]          = df_indices

if __name__ == "__main__":
    main()
##%% QUITAR UN HASHTAG PARA PROBRAR EL BLOQUE
# ============================================================
# CELDA - BACKTEST FINAL DEL MODELO GA v2
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="notebook")

# ------------------------------------------------------------
# 1. Comprobaciones y rutas
# ------------------------------------------------------------
if "best_global_genome" not in globals() or best_global_genome is None:
    if "RUTA_MEJOR_GENOMA" in globals() and Path(RUTA_MEJOR_GENOMA).exists():
        best_global_genome = np.load(RUTA_MEJOR_GENOMA)
        print(f"Genoma cargado desde disco: {RUTA_MEJOR_GENOMA}")
    else:
        raise RuntimeError("No existe best_global_genome en memoria ni en disco.")

BACKTEST_DIR = DIR_SALIDA / "backtest_results"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. Funciones auxiliares para backtest
# ------------------------------------------------------------
def calc_extended_metrics(date_series, capital_series, rf_annual=0.02):
    s = pd.DataFrame({
        "Date"   : pd.to_datetime(np.asarray(date_series)),
        "Capital": np.asarray(capital_series, dtype=float),
    }).dropna().copy()
    s = s.sort_values("Date").drop_duplicates("Date")
    s["returns"] = s["Capital"].pct_change().fillna(0.0)

    if len(s) < 2:
        return {k: np.nan for k in [
            "Total Return (%)", "Annualized Return (%)", "Volatility (%)",
            "Sharpe Ratio", "Sortino Ratio", "Max Drawdown (%)",
            "Calmar Ratio", "Win Rate (%)", "Final Value (€)", "Period (years)",
        ]}

    total_return      = (s["Capital"].iloc[-1] / s["Capital"].iloc[0] - 1) * 100
    n_days            = (s["Date"].iloc[-1] - s["Date"].iloc[0]).days
    years             = n_days / 365.25 if n_days > 0 else np.nan
    annualized_return = (
        ((s["Capital"].iloc[-1] / s["Capital"].iloc[0]) ** (365.25 / n_days) - 1) * 100
        if n_days > 0 else np.nan
    )
    volatility        = s["returns"].std(ddof=0) * np.sqrt(252) * 100
    rf_daily          = (1 + rf_annual) ** (1 / 252) - 1
    excess_returns    = s["returns"] - rf_daily
    sharpe_ratio      = np.sqrt(252) * excess_returns.mean() / (s["returns"].std(ddof=0) + 1e-12)
    cumulative        = (1 + s["returns"]).cumprod()
    running_max       = cumulative.cummax()
    drawdown          = cumulative / running_max - 1
    max_drawdown      = drawdown.min() * 100
    negative_returns  = s.loc[s["returns"] < 0, "returns"]
    downside_std      = negative_returns.std(ddof=0) * np.sqrt(252) if len(negative_returns) > 0 else np.nan
    sortino_ratio     = (
        ((annualized_return / 100) - rf_annual) / (downside_std + 1e-12)
        if pd.notnull(downside_std) else np.nan
    )
    calmar_ratio      = (
        (annualized_return / abs(max_drawdown))
        if pd.notnull(max_drawdown) and max_drawdown < 0 else np.nan
    )
    win_rate          = (s["returns"] > 0).sum() / len(s["returns"]) * 100

    return {
        "Total Return (%)"     : total_return,
        "Annualized Return (%)": annualized_return,
        "Volatility (%)"       : volatility,
        "Sharpe Ratio"         : sharpe_ratio,
        "Sortino Ratio"        : sortino_ratio,
        "Max Drawdown (%)"     : max_drawdown,
        "Calmar Ratio"         : calmar_ratio,
        "Win Rate (%)"         : win_rate,
        "Final Value (€)"      : s["Capital"].iloc[-1],
        "Period (years)"       : years,
    }

def format_metrics_df(metrics_dict):
    dfm    = pd.DataFrame(metrics_dict).T
    df_fmt = dfm.copy()

    pct_cols = ["Total Return (%)", "Annualized Return (%)", "Volatility (%)",
                "Max Drawdown (%)", "Win Rate (%)"]
    num_cols = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Period (years)"]
    eur_cols = ["Final Value (€)"]

    for c in pct_cols:
        if c in df_fmt.columns:
            df_fmt[c] = df_fmt[c].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "NaN")
    for c in num_cols:
        if c in df_fmt.columns:
            df_fmt[c] = df_fmt[c].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "NaN")
    for c in eur_cols:
        if c in df_fmt.columns:
            df_fmt[c] = df_fmt[c].apply(lambda x: f"€{x:,.2f}" if pd.notnull(x) else "NaN")

    return dfm, df_fmt

def run_ga_backtest_episode(env, policy, genome):
    obs, info = env.reset()
    done      = False

    fechas, net_worths, rewards_list = [], [], []
    cash_weights, turnovers, transaction_costs = [], [], []
    weights_history, actions_history          = [], []
    ibex_log_rets                             = []

    while not done:
        action = policy.forward(obs, genome)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

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
        "Date"           : fechas,
        "GA Agent"       : net_worths,
        "Reward"         : rewards_list,
        "CashWeight"     : cash_weights,
        "Turnover"       : turnovers,
        "TransactionCost": transaction_costs,
        "IBEX_LogRet"    : ibex_log_rets,
    }).drop_duplicates("Date").sort_values("Date")

    return df_path, weights_history, actions_history

def calculate_equal_weight_bt(df_test, fechas_bt, capital_inicial):
    fechas_bt  = pd.to_datetime(fechas_bt)
    df_sorted  = df_test.sort_values(["Ticker", "Date"]).copy()
    df_sorted["Retorno_Diario"] = (
        df_sorted.groupby("Ticker")[DEFAULT_PARAMS["TARGET_PRICE_COL"]].pct_change()
    )
    df_bench    = df_sorted[df_sorted["Date"].isin(fechas_bt)].copy()
    retornos_ew = (
        df_bench.groupby("Date")["Retorno_Diario"]
        .mean().reindex(fechas_bt).fillna(0.0)
    )
    ew_net_worth = capital_inicial * (1 + retornos_ew).cumprod()
    if len(ew_net_worth) > 0:
        ew_net_worth.iloc[0] = capital_inicial
    return pd.DataFrame({"Date": fechas_bt, "Equal Weight": ew_net_worth.values})

def calculate_buyhold_bt(df_test, fechas_bt, capital_inicial):
    fechas_bt = pd.to_datetime(fechas_bt)
    df_prices = (
        df_test.pivot(index="Date", columns="Ticker",
                      values=DEFAULT_PARAMS["TARGET_PRICE_COL"])
        .sort_index()
    )
    df_prices_aligned = (
        df_prices.reindex(df_prices.index.union(fechas_bt))
        .sort_index().ffill().reindex(fechas_bt)
    )
    df_prices_aligned = df_prices_aligned.dropna(axis=1, how="any")

    if df_prices_aligned.empty:
        return pd.DataFrame({"Date": fechas_bt, "Buy & Hold": np.full(len(fechas_bt), capital_inicial)})

    n_assets       = df_prices_aligned.shape[1]
    initial_prices = df_prices_aligned.iloc[0].values
    shares_bought  = (capital_inicial / n_assets) / initial_prices
    portfolio_vals = (df_prices_aligned.values * shares_bought).sum(axis=1)
    return pd.DataFrame({"Date": fechas_bt, "Buy & Hold": portfolio_vals})

def calculate_indices_bt(df_indices_arg, fechas_bt, capital_inicial):
    fechas_bt = pd.to_datetime(fechas_bt)
    df_idx    = df_indices_arg.copy()
    df_idx    = df_idx.dropna(subset=["Date"]).drop_duplicates(
        subset=["Date", "Ticker"], keep="last"
    ).sort_values(["Ticker", "Date"])

    df_idx_pivot   = df_idx.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    df_idx_aligned = (
        df_idx_pivot.reindex(df_idx_pivot.index.union(fechas_bt))
        .sort_index().ffill().reindex(fechas_bt)
    )
    df_idx_aligned = df_idx_aligned.dropna(axis=1, how="any")

    out = {}
    for col in df_idx_aligned.columns:
        ret = df_idx_aligned[col].pct_change().fillna(0.0)
        nw  = capital_inicial * (1 + ret).cumprod()
        if len(nw) > 0:
            nw.iloc[0] = capital_inicial
        out[col] = nw.values
    return out

def infer_trades_from_weights(df_weights, min_delta=0.01):
    if df_weights.empty or len(df_weights) < 2:
        return pd.DataFrame()

    asset_cols = [c for c in df_weights.columns if c != "Date"]
    trades     = []
    trade_id   = 1
    prev       = df_weights.iloc[0]

    for i in range(1, len(df_weights)):
        curr = df_weights.iloc[i]
        dt   = curr["Date"]

        for asset in asset_cols:
            old_w = float(prev.get(asset, 0.0)) if pd.notnull(prev.get(asset, np.nan)) else 0.0
            new_w = float(curr.get(asset, 0.0)) if pd.notnull(curr.get(asset, np.nan)) else 0.0
            delta = new_w - old_w

            if abs(delta) >= min_delta:
                trades.append({
                    "Trade_ID"       : trade_id,
                    "Date"           : dt,
                    "Ticker"         : asset,
                    "Trade_Type"     : "BUY" if delta > 0 else "SELL",
                    "Old_Weight"     : old_w,
                    "New_Weight"     : new_w,
                    "Delta_Weight"   : delta,
                    "Abs_Delta_Weight": abs(delta),
                })
                trade_id += 1

        prev = curr.copy()
    return pd.DataFrame(trades)

# ------------------------------------------------------------
# 3. Entorno de backtest (2024-2025)
# ------------------------------------------------------------
env_backtest = make_env(
    df         = df_backtest,
    tickers    = tickers,
    params     = DEFAULT_PARAMS,
    feature_cols= feature_cols,
    df_indices = df_indices,
    seed       = int(DEFAULT_PARAMS["SEED"]),
)

# ------------------------------------------------------------
# 4. Backtest del agente GA
# ------------------------------------------------------------
df_ga, weights_history, actions_history = run_ga_backtest_episode(
    env    = env_backtest,
    policy = policy,
    genome = best_global_genome,
)

fechas_bt      = df_ga["Date"].values
capital_inicial = float(DEFAULT_PARAMS["CAPITAL_INICIAL"])

df_ew  = calculate_equal_weight_bt(df_backtest, fechas_bt, capital_inicial)
df_bh  = calculate_buyhold_bt(df_backtest, fechas_bt, capital_inicial)
idx_dict = calculate_indices_bt(df_indices, fechas_bt, capital_inicial)

# ------------------------------------------------------------
# 5. Comparación de estrategias
# ------------------------------------------------------------
comparison_df = pd.DataFrame({"Date": fechas_bt})
comparison_df = comparison_df.merge(df_ga[["Date", "GA Agent"]], on="Date", how="left")
comparison_df = comparison_df.merge(df_bh,                       on="Date", how="left")
comparison_df = comparison_df.merge(df_ew,                       on="Date", how="left")

for k, v in idx_dict.items():
    comparison_df[k] = v

comparison_df = comparison_df.sort_values("Date").reset_index(drop=True)

comparison_norm = comparison_df.copy()
for c in [x for x in comparison_norm.columns if x != "Date"]:
    if comparison_norm[c].iloc[0] != 0:
        comparison_norm[c] = (comparison_norm[c] / comparison_norm[c].iloc[0]) * 100

# ------------------------------------------------------------
# 6. Métricas extendidas del backtest
# ------------------------------------------------------------
backtest_results = {}
for c in [x for x in comparison_df.columns if x != "Date"]:
    backtest_results[c] = calc_extended_metrics(
        comparison_df["Date"].values,
        comparison_df[c].values,
    )

metrics_num, metrics_fmt = format_metrics_df(backtest_results)

# ------------------------------------------------------------
# 7. Historial de pesos y trades
# ------------------------------------------------------------
weights_cols = tickers + ["CASH"]
weights_rows = []

for dt, w in zip(fechas_bt, weights_history):
    if len(w) == len(weights_cols):
        row = {"Date": dt}
        for col, val in zip(weights_cols, w):
            row[col] = float(val)
        weights_rows.append(row)

df_weights = pd.DataFrame(weights_rows)
df_trades  = (
    infer_trades_from_weights(df_weights, min_delta=0.02) #PARAMETRO A MODIFICAR
    if not df_weights.empty else pd.DataFrame()
)

# ------------------------------------------------------------
# 8. Guardado de resultados
# ------------------------------------------------------------
comparison_df.to_csv(  BACKTEST_DIR / "portfolio_evolution.csv",         index=False)
comparison_norm.to_csv(BACKTEST_DIR / "portfolio_evolution_base100.csv",  index=False)
df_ga.to_csv(          BACKTEST_DIR / "ga_path_details.csv",              index=False)
metrics_num.to_csv(    BACKTEST_DIR / "performance_metrics.csv")

if not df_weights.empty:
    df_weights.to_csv(BACKTEST_DIR / "portfolio_composition_history.csv", index=False)
if not df_trades.empty:
    df_trades.to_csv( BACKTEST_DIR / "all_trades.csv",                    index=False)

with open(BACKTEST_DIR / "reporte_metricas.txt", "w", encoding="utf-8") as f:
    f.write("REPORTE DE BACKTEST - GA MULTISTOCK v2\n")
    f.write("=" * 80 + "\n\n")
    f.write(metrics_fmt.to_string())

# ------------------------------------------------------------
# 9. Gráficos
# ------------------------------------------------------------
# Gráfico 1: evolución absoluta
plt.figure(figsize=(16, 8))
for c in [x for x in comparison_df.columns if x != "Date"]:
    lw = 2.5 if c == "GA Agent" else 1.5
    ls = "-"   if c == "GA Agent" else "--"
    plt.plot(comparison_df["Date"], comparison_df[c], label=c, linewidth=lw, linestyle=ls)
plt.title("Backtest 2024-2025 — Evolución del valor de cartera", fontsize=14, fontweight="bold")
plt.legend(fontsize=9, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BACKTEST_DIR / "grafico_comparativo.png", dpi=300, bbox_inches="tight")
plt.show()

# Gráfico 2: base 100
plt.figure(figsize=(16, 8))
for c in [x for x in comparison_norm.columns if x != "Date"]:
    lw = 2.5 if c == "GA Agent" else 1.5
    ls = "-"   if c == "GA Agent" else "--"
    plt.plot(comparison_norm["Date"], comparison_norm[c], label=c, linewidth=lw, linestyle=ls)
plt.axhline(100, color="black", linestyle=":", alpha=0.6)
plt.title("Backtest 2024-2025 — Evolución normalizada base 100", fontsize=14, fontweight="bold")
plt.legend(fontsize=9, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BACKTEST_DIR / "grafico_base100.png", dpi=300, bbox_inches="tight")
plt.show()

# Gráfico 3: turnover y cash
plt.figure(figsize=(16, 5))
plt.plot(df_ga["Date"], df_ga["Turnover"] * 100, label="Turnover (%)", color="black")
plt.fill_between(df_ga["Date"], 0, df_ga["CashWeight"] * 100,
                 alpha=0.25, color="red", label="Cash Weight (%)")
plt.title("Turnover y peso en cash — Backtest 2024-2025", fontsize=13, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BACKTEST_DIR / "turnover_cash.png", dpi=300, bbox_inches="tight")
plt.show()

# Gráfico 4: reward acumulada vs IBEX 35
df_ga_sorted = df_ga.sort_values("Date").copy()
df_ga_sorted["Reward_Acum"] = df_ga_sorted["Reward"].cumsum()
ibex_logret_acum = df_ga_sorted["IBEX_LogRet"].cumsum()

plt.figure(figsize=(16, 5))
plt.plot(df_ga_sorted["Date"], df_ga_sorted["Reward_Acum"],
         label="Reward acumulada (α-IBEX)", color="seagreen", linewidth=2)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Diferencial acumulado log-ret cartera − IBEX 35 — Backtest 2024-2025",
          fontsize=13, fontweight="bold")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BACKTEST_DIR / "reward_acumulada.png", dpi=300, bbox_inches="tight")
plt.show()

# Gráfico 5: composición de cartera (si hay historial)
if not df_weights.empty:
    df_plot = df_weights.set_index("Date")

    plt.figure(figsize=(18, 8))
    plt.stackplot(df_plot.index, df_plot.T.values, labels=df_plot.columns, alpha=0.85)
    plt.legend(loc="upper left", ncol=4, fontsize=7)
    plt.title("Evolución de la composición de cartera — Backtest 2024-2025",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(BACKTEST_DIR / "portfolio_composition_evolution.png", dpi=300, bbox_inches="tight")
    plt.show()

    last_row = df_plot.iloc[-1]
    last_row = last_row[last_row > 0.01].sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(last_row.index, last_row.values * 100,
                    color="steelblue", edgecolor="black", alpha=0.85)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{width:.2f}%", va="center")
    plt.title("Composición final de la cartera — Backtest 2024-2025",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(BACKTEST_DIR / "composicion_final_barras.png", dpi=300, bbox_inches="tight")
    plt.show()

# ── Gráfico 7: Drawdown del GA Agent vs IBEX 35 ─────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
for col, color, lw in [("GA Agent", "steelblue", 2), ("IBEX 35", "tomato", 1.5)]:
    if col in comparison_df.columns:
        series = comparison_df[col].values.astype(float)
        running_max = np.maximum.accumulate(np.where(np.isfinite(series), series, np.nan))
        dd = (series / running_max - 1) * 100
        ax.fill_between(comparison_df["Date"], dd, 0, alpha=0.25, color=color)
        ax.plot(comparison_df["Date"], dd, label=col, color=color, linewidth=lw)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Drawdown — GA Agent vs IBEX 35 (Backtest 2024–2025)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Drawdown (%)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BACKTEST_DIR / "drawdown.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Gráfico 8: Distribución de retornos diarios ──────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for col, color in [("GA Agent", "steelblue"), ("IBEX 35", "tomato")]:
    if col in comparison_df.columns:
        rets = comparison_df[col].pct_change().dropna() * 100
        ax.hist(rets, bins=60, alpha=0.45, color=color, label=col, edgecolor="none")
        ax.axvline(rets.mean(), color=color, linestyle="--", linewidth=1.2)
ax.set_title("Distribución de retornos diarios (Backtest 2024–2025)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Retorno diario (%)")
ax.set_ylabel("Frecuencia")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(BACKTEST_DIR / "distribucion_retornos.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Gráfico 9: Evolución del fitness por generación ──────────────────
_ruta_ga_results = RUTA_RESULTADOS
if _ruta_ga_results.exists():
    df_gen = pd.read_csv(_ruta_ga_results)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_gen["generation"], df_gen["best_score"],  marker="o", label="Mejor score", color="steelblue", linewidth=2)
    ax.plot(df_gen["generation"], df_gen["mean_score"],  marker="s", label="Score medio",  color="gray",      linewidth=1.5, linestyle="--")
    ax.fill_between(
        df_gen["generation"],
        df_gen["mean_score"] - df_gen["std_score"],
        df_gen["mean_score"] + df_gen["std_score"],
        alpha=0.15, color="gray"
    )
    ax.set_title("Evolución del fitness por generación (entrenamiento)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(BACKTEST_DIR / "fitness_generaciones.png", dpi=150, bbox_inches="tight")
    plt.show()

# ── Gráfico 10: Tabla resumen de métricas ────────────────────────────
fig, ax = plt.subplots(figsize=(14, len(metrics_fmt) * 0.55 + 1.5))
ax.axis("off")
col_labels = list(metrics_fmt.columns)
row_labels  = list(metrics_fmt.index)
cell_text   = metrics_fmt.values.tolist()
tbl = ax.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=col_labels,
    cellLoc="center",
    rowLoc="center",
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.6)
# Resaltar fila GA Agent
for j in range(len(col_labels)):
    tbl[(row_labels.index("GA Agent") + 1, j)].set_facecolor("#d6eaf8")
ax.set_title("Resumen de métricas — Backtest 2024–2025",
             fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(BACKTEST_DIR / "tabla_metricas.png", dpi=150, bbox_inches="tight")
plt.show()

# ── GRÁFICO: Correlación de retornos de la cartera final ──────────────────────
# Requiere: dfweights (portfolio_composition_history), dfbacktest (dataset backtest)

# Extraer tickers con peso > 0 en el último día (cartera final activa)
asset_cols = [c for c in df_weights.columns if c != 'Date' and c != 'CASH']
last_weights = df_weights.iloc[-1][asset_cols]
active_tickers = last_weights[last_weights > 0.005].index.tolist()  # umbral 0.5%

if len(active_tickers) >= 2:
    # Pivotar precios de cierre del periodo backtest
    df_prices_bt = df_backtest[df_backtest['Ticker'].isin(active_tickers)].copy()
    df_pivot = (df_prices_bt
                .pivot(index='Date', columns='Ticker', values=DEFAULT_PARAMS['TARGET_PRICE_COL'])
                .sort_index()
                .ffill())
    df_returns = df_pivot.pct_change().dropna()
    corr_matrix = df_returns.corr()

    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(active_tickers)
    fig_size = max(10, int(n * 0.6))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # triángulo superior
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True if n <= 15 else False,
        fmt='.2f',
        cmap='RdBu',
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.4,
        square=True,
        ax=ax,
        cbar_kws={'shrink': 0.7, 'label': 'Correlación de Pearson'}
    )
    ax.set_title(
        f'Correlación de retornos — cartera final ({len(active_tickers)} activos)\n'
        f'Backtest {DEFAULT_PARAMS["ANIO_INICIO_BACKTEST"]}–{DEFAULT_PARAMS["ANIO_FIN_BACKTEST"]}',
        fontsize=13, fontweight='bold', pad=14
    )
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    plt.tight_layout()
    plt.savefig(BACKTEST_DIR / 'correlacion_cartera_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✔ Correlación guardada → {BACKTEST_DIR / 'correlacion_cartera_final.png'}")
else:
    print("⚠ Menos de 2 activos activos en la cartera final. No se genera correlación.")
    
# ── GRÁFICO: Composición de la cartera final por Sector ───────────────────────
# Requiere: dfweights, dfbacktest (con columna 'Sector')

if 'Sector' in df_backtest.columns:
    # Mapa ticker → sector (tomar el sector más frecuente por ticker)
    sector_map = (df_backtest[['Ticker', 'Sector']]
                  .dropna()
                  .groupby('Ticker')['Sector']
                  .agg(lambda x: x.mode()[0])
                  .to_dict())

    # Pesos finales de la cartera (último día, sin CASH)
    last_w = df_weights.iloc[-1].copy()
    last_date = df_weights.iloc[-1]['Date']
    peso_activos = {t: last_w[t] for t in asset_cols if last_w.get(t, 0) > 0.005}

    # Agrupar pesos por sector
    sector_weights = {}
    sin_sector = 0.0
    for ticker, peso in peso_activos.items():
        sector = sector_map.get(ticker, None)
        if sector:
            sector_weights[sector] = sector_weights.get(sector, 0.0) + peso
        else:
            sin_sector += peso
    if sin_sector > 0:
        sector_weights['Sin clasificar'] = sin_sector

    # Ordenar de mayor a menor peso
    sector_df = (pd.DataFrame.from_dict(sector_weights, orient='index', columns=['Peso'])
                 .sort_values('Peso', ascending=False))
    sector_df['Peso_pct'] = sector_df['Peso'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Barras horizontales ---
    ax1 = axes[0]
    colors = plt.cm.tab20.colors[:len(sector_df)]
    bars = ax1.barh(sector_df.index, sector_df['Peso_pct'], color=colors, edgecolor='black', alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                 f'{w:.1f}%', va='center', fontsize=9)
    ax1.set_xlabel('Peso en cartera (%)')
    ax1.set_title(f'Composición por Sector — {pd.Timestamp(last_date).strftime("%Y-%m-%d")}',
                  fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # --- Pie chart ---
    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        sector_df['Peso_pct'],
        labels=sector_df.index,
        autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
        colors=colors,
        startangle=140,
        wedgeprops=dict(edgecolor='white', linewidth=1.2)
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax2.set_title('Distribución sectorial (%)', fontsize=12, fontweight='bold')

    plt.suptitle(
        f'Cartera final GA — Backtest {DEFAULT_PARAMS["ANIO_INICIO_BACKTEST"]}–{DEFAULT_PARAMS["ANIO_FIN_BACKTEST"]}',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(BACKTEST_DIR/ 'composicion_sectorial.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✔ Composición sectorial guardada → {BACKTEST_DIR / 'composicion_sectorial.png'}")

    # También exportar tabla de sectores a CSV
    sector_df.to_csv(BACKTEST_DIR / 'composicion_sectorial.csv')
else:
    print("⚠ La columna 'Sector' no está disponible en dfbacktest.")
    
"""   
YA NO SE USA ESTE BLOQUE  
# ── ANÁLISIS: Cruce cartera inicial vs cartera final del GA ───────────────────
# Requiere: dfweights, RUTA_PORTFOLIO

df_inicial = pd.read_csv(RUTA_PORTFOLIO)

# Normalizar nombre columna de pesos iniciales
peso_col_ini = next((c for c in df_inicial.columns
                     if c.strip().lower() in ['peso_inicial', 'weight', 'peso', 'weight_target']), None)
ticker_col_ini = next((c for c in df_inicial.columns
                       if c.strip().lower() in ['ticker', 'ric', 'symbol']), None)

if peso_col_ini and ticker_col_ini:
    df_inicial = df_inicial[[ticker_col_ini, peso_col_ini]].copy()
    df_inicial.columns = ['Ticker', 'Peso_Inicial']
    df_inicial['Peso_Inicial'] = pd.to_numeric(df_inicial['Peso_Inicial'], errors='coerce').fillna(0.0)
    tickers_ini = set(df_inicial[df_inicial['Peso_Inicial'] > 0]['Ticker'].tolist())

    # Cartera final del GA (último día, sin CASH)
    last_w_series = df_weights.iloc[-1]
    tickers_fin = {t: last_w_series[t] for t in asset_cols if last_w_series.get(t, 0) > 0.005}
    tickers_fin_set = set(tickers_fin.keys())

    # Clasificación
    mantenidas  = tickers_ini & tickers_fin_set          # estaban y siguen
    eliminadas  = tickers_ini - tickers_fin_set           # estaban y se han quitado
    incorporadas = tickers_fin_set - tickers_ini          # no estaban y se han añadido

    # --- Tabla resumen ---
    rows = []
    for t in sorted(mantenidas):
        peso_i = df_inicial[df_inicial['Ticker'] == t]['Peso_Inicial'].values
        peso_i = float(peso_i[0]) if len(peso_i) > 0 else 0.0
        rows.append({'Ticker': t, 'Estado': '✅ Mantenida',
                     'Peso_Inicial_%': round(peso_i * 100, 2),
                     'Peso_Final_%':   round(tickers_fin[t] * 100, 2),
                     'Δ_pp': round((tickers_fin[t] - peso_i) * 100, 2)})
    for t in sorted(eliminadas):
        peso_i = df_inicial[df_inicial['Ticker'] == t]['Peso_Inicial'].values
        peso_i = float(peso_i[0]) if len(peso_i) > 0 else 0.0
        rows.append({'Ticker': t, 'Estado': '❌ Eliminada',
                     'Peso_Inicial_%': round(peso_i * 100, 2),
                     'Peso_Final_%':   0.0,
                     'Δ_pp': round(-peso_i * 100, 2)})
    for t in sorted(incorporadas):
        rows.append({'Ticker': t, 'Estado': '🆕 Incorporada',
                     'Peso_Inicial_%': 0.0,
                     'Peso_Final_%':   round(tickers_fin[t] * 100, 2),
                     'Δ_pp': round(tickers_fin[t] * 100, 2)})

    df_cruce = pd.DataFrame(rows).sort_values(['Estado', 'Ticker'])
    df_cruce.to_csv(BACKTEST_DIR / 'cruce_cartera_inicial_final.csv', index=False)

    # --- Gráfico: Waterfall / barras comparativas ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Subplot izquierdo: Venn numérico (barras apiladas)
    categorias = ['Mantenidas', 'Eliminadas', 'Incorporadas']
    counts = [len(mantenidas), len(eliminadas), len(incorporadas)]
    colores = ['#2ecc71', '#e74c3c', '#3498db']
    bars = axes[0].bar(categorias, counts, color=colores, edgecolor='black', alpha=0.85, width=0.5)
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(count), ha='center', va='bottom', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Número de acciones')
    axes[0].set_title('Cambios en cartera: inicial → final', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, max(counts) + 2)
    axes[0].grid(axis='y', alpha=0.3)

    # Anotación de tasa de renovación
    tasa = len(eliminadas) / len(tickers_ini) * 100 if tickers_ini else 0
    axes[0].text(0.98, 0.97,
                 f'Tasa de rotación: {tasa:.0f}%\n'
                 f'Inicial: {len(tickers_ini)} acciones\n'
                 f'Final:   {len(tickers_fin_set)} acciones',
                 transform=axes[0].transAxes, ha='right', va='top',
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Subplot derecho: comparativa de pesos (mantenidas)
    if mantenidas:
        df_mant = df_cruce[df_cruce['Estado'] == '✅ Mantenida'].sort_values('Δ_pp')
        x = np.arange(len(df_mant))
        w = 0.35
        axes[1].bar(x - w/2, df_mant['Peso_Inicial_%'], w, label='Inicial', color='#95a5a6', alpha=0.85)
        axes[1].bar(x + w/2, df_mant['Peso_Final_%'],   w, label='Final GA', color='#2ecc71', alpha=0.85)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df_mant['Ticker'], rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('Peso en cartera (%)')
        axes[1].set_title('Pesos inicial vs final — acciones mantenidas', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No hay acciones mantenidas', ha='center', va='center',
                     fontsize=12, transform=axes[1].transAxes)

    plt.suptitle('Cruce cartera inicial (portfolio.inicial.csv) vs cartera final GA',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(BACKTEST_DIR / 'cruce_inicial_final.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Imprimir resumen en consola ---
    print("\n" + "=" * 60)
    print(f"CRUCE CARTERA INICIAL → FINAL")
    print("=" * 60)
    print(f"  ✅ Mantenidas  ({len(mantenidas):2d}): {sorted(mantenidas)}")
    print(f"  ❌ Eliminadas  ({len(eliminadas):2d}): {sorted(eliminadas)}")
    print(f"  🆕 Incorporadas({len(incorporadas):2d}): {sorted(incorporadas)}")
    print(f"\n  Tasa de rotación estructural: {tasa:.1f}%")
    print(f"\nTabla completa guardada → {BACKTEST_DIR / 'cruce_cartera_inicial_final.csv'}")
    print(f"Gráfico guardado        → {BACKTEST_DIR / 'cruce_inicial_final.png'}")

else:
    print("⚠ No se encontró columna Ticker/Peso en portfolio.inicial.csv. "
          f"Columnas disponibles: {df_inicial.columns.tolist()}")
"""
# ------------------------------------------------------------
# 10. Mostrar métricas
# ------------------------------------------------------------
print("\n========== MÉTRICAS DE BACKTEST 2024-2025 ==========\n")
try:
    display(metrics_fmt)
except NameError:
    print(metrics_fmt.to_string())

print(f"\nResultados guardados en: {BACKTEST_DIR}")

elapsed_total = time.perf_counter() - _t_inicio_total
print(f"\n⏱ Tiempo total de ejecución: {elapsed_total/60:.1f} min ({elapsed_total:.0f}s)")

"""
# %% CHUNK DE RECUPERACIÓN DE DATOS Y MODELO
# Ejecuta esta celda sola para cargar todo en memoria sin reentrenar

# 1. Cargamos índices y datos usando tus funciones ya creadas
df_indices = load_indices()
(df_train, df_backtest, tickers_local, feature_cols_local, schema, null_report) = prepare_data(DEFAULT_PARAMS)

# 2. Cargamos el genoma ganador que se guardó a salvo en el disco duro
best_global_genome = np.load(RUTA_MEJOR_GENOMA)

# 3. Recreamos la estructura del agente (Policy)
env_fake = make_env(df_train, tickers_local, DEFAULT_PARAMS, feature_cols_local, df_indices)
policy = GeneticMLPPolicy(
    obs_dim    = env_fake.observation_space.shape[0],
    action_dim = env_fake.action_space.shape[0],
    hidden_dim = int(DEFAULT_PARAMS["hidden_dim"]),
)

# 4. Renombramos las variables para que coincidan con lo que pide tu código de backtest
tickers = tickers_local
feature_cols = feature_cols_local

print("✅ Todo cargado en memoria. El agente está listo.")
"""