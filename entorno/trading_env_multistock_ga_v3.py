"""
trading_env_multistock_ga_2.py

Entorno de trading multi-activo para Algoritmo Genético (v2).

Diferencias respecto a MultiStockTradingEnvGA (v1):
- Composición de cartera inicial leída de portfolio/portfolio_inicial.csv
  (columna de pesos: Peso_Inicial).
- El algoritmo puede operar con TODAS las acciones del universo (sin límite
  de cardinalidad, aunque se mantiene max_positions como parámetro opcional).
- La composición total de la cartera NUNCA puede superar el 100 %:
  los pesos se normalizan siempre a suma ≤ 1 (el resto queda como CASH).
- Función de recompensa: diferencial diario del retorno logarítmico de la
  cartera menos el retorno logarítmico del IBEX 35.
  reward_t = log(NW_t / NW_{t-1}) - log(IBEX_t / IBEX_{t-1})
- El CSV de índices (indices_cierre.csv) se carga externamente y se pasa
  al constructor; se usa únicamente el Ticker == 'IBEX 35'.

Rutas de directorio conservadas del script original:
  - portfolio_init_path : "portfolio/portfolio.inicial.csv" YA NO SE USA, EMPIEZA EN CASH
  - El dataset de entrenamiento se pasa como DataFrame (df_train).
  - El CSV de índices se pasa como DataFrame (df_indices).
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import warnings

# ──────────────────────────────────────────────────────────────────────────────
# Constantes globales (iguales al entorno v1)
# ──────────────────────────────────────────────────────────────────────────────
CAPITAL_INICIAL        = 100_000.0
COMISION_BROKER        = 0.001
TASA_LIBRE_RIESGO      = 0.02
VENTANA_HISTORICA      = 10
FRECUENCIA_REBALANCEO  = 5
CASTIGO_BANCARROTA     = -100.0
UMBRAL_QUIEBRA         = 0.10
MAX_ACTIVOS_EN_CARTERA = 10   # opcional; None = sin límite


# ──────────────────────────────────────────────────────────────────────────────
# Helper: construcción robusta del DataFrame de entrenamiento
# ──────────────────────────────────────────────────────────────────────────────
def build_df_train(df, start_year=None, end_year=None, tickers=None):
    """
    Construye df_train de forma explícita y robusta.
    - Convierte Date.
    - Ordena por Date y Ticker.
    - Filtra por años si se indica.
    - Filtra por universo de tickers si se indica.
    """
    out = df.copy()
    if "Date" not in out.columns or "Ticker" not in out.columns:
        raise ValueError("El DataFrame debe contener al menos las columnas 'Date' y 'Ticker'.")

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", "Ticker"]).copy()

    if start_year is not None:
        out = out[out["Date"].dt.year >= int(start_year)].copy()
    if end_year is not None:
        out = out[out["Date"].dt.year <= int(end_year)].copy()
    if tickers is not None:
        out = out[out["Ticker"].isin(list(tickers))].copy()

    out = out.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    if out.empty:
        raise ValueError("df_train queda vacío tras aplicar los filtros indicados.")

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Entorno principal
# ──────────────────────────────────────────────────────────────────────────────
class MultiStockTradingEnvGA2(gym.Env):
    """
    Entorno multiactivo para Algoritmo Genético (v2).

    Reglas implementadas:
    ─────────────────────
    1. Parte de una cartera inicial definida en portfolio/portfolio.inicial.csv
       (columna de pesos: Peso_Inicial).
    2. Puede observar y operar con TODAS las acciones del universo.
    3. La composición total NUNCA supera el 100 % (la suma de pesos de acciones
       + cash = 1.0 siempre).
    4. Función de recompensa = diferencial diario del log-retorno de la cartera
       menos el log-retorno del IBEX 35.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df_train,
        tickers,
        df_indices,                              # DataFrame con indices_cierre.csv
        features=None,
        initial_balance=CAPITAL_INICIAL,
        commission=COMISION_BROKER,
        risk_free_rate=TASA_LIBRE_RIESGO,
        window_size=VENTANA_HISTORICA,
        rebalance_freq=FRECUENCIA_REBALANCEO,
        obs_clip=VENTANA_HISTORICA,
        max_positions=10,                      # None = sin límite de cardinalidad
        portfolio_init_path="portfolio/portfolio_inicial.csv",
        
    ):
        super().__init__()

        # ── Universo de activos ───────────────────────────────────────────────
        self.df       = build_df_train(df_train, tickers=tickers)
        self.tickers  = sorted(list(tickers))
        self.num_stocks  = len(self.tickers)
        self.num_assets  = self.num_stocks + 1          # acciones + CASH

        # ── Parámetros ────────────────────────────────────────────────────────
        self.initial_balance     = float(initial_balance)
        self.commission          = float(commission)
        self.window_size         = int(window_size)
        self.rebalance_freq      = int(rebalance_freq)
        self.obs_clip            = float(obs_clip)
        self.max_positions       = 10        # None = sin límite
        self.risk_free_rate_daily = (1.0 + float(risk_free_rate)) ** (1.0 / 252.0) - 1.0

        # ── Ruta cartera inicial ──────────────────────────────────────────────
        self.portfolio_init_path = Path(portfolio_init_path)

        # ── Índice de referencia (IBEX 35) ────────────────────────────────────
        self._load_benchmark(df_indices)

        # ── Preparación del dataset ───────────────────────────────────────────
        # ── Preparación del dataset ───────────────────────────────────────────
        self._prepare_columns(features)
        self._validate_data()
        self._prepare_dataframe()
        self._build_tensor_cache()
        
        # ── Forzar cartera inicial 100% CASH ──
        self.initial_target_weights = np.zeros(self.num_assets, dtype=np.float64)
        self.initial_target_weights[-1] = 1.0

        # ── Espacios de observación y acción ─────────────────────────────────
        obs_shape = (
            self.window_size * self.num_stocks * len(self.features)
            + self.num_assets  # pesos actuales (acciones + cash)
            + 5                # drawdown, last_ret, last_turnover, cash_ratio,
                               # ibex_log_ret (contexto del benchmark del día anterior)
        )
        # Acciones: logits sin restricción de signo → softmax los convierte a
        # pesos válidos que siempre suman 1 (incluye CASH).
        self.action_space      = spaces.Box(
            low=-10.0, high=10.0, shape=(self.num_assets,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )

        self.last_info = None
        self.reset()

    # ──────────────────────────────────────────────────────────────────────────
    # Benchmark IBEX 35
    # ──────────────────────────────────────────────────────────────────────────
    def _load_benchmark(self, df_indices):
        """
        Carga el IBEX 35 desde df_indices y precalcula los log-retornos en un diccionario temporal.
        Posteriormente, `_build_tensor_cache` convertirá esto en un array Numpy ultra-rápido.
        """
        df_idx = df_indices.copy()
        df_idx["Date"]  = pd.to_datetime(df_idx["Date"], errors="coerce").dt.normalize()
        df_idx["Close"] = pd.to_numeric(df_idx["Close"], errors="coerce")
        df_idx          = df_idx.dropna(subset=["Date", "Close"])

        ibex = (
            df_idx[df_idx["Ticker"].str.strip() == "IBEX 35"]
            .sort_values("Date")
            .drop_duplicates("Date")
            .reset_index(drop=True)
        )

        if ibex.empty:
            raise ValueError("No se encontró el ticker 'IBEX 35' en df_indices.")

        ibex["log_ret_ibex"] = np.log(ibex["Close"] / ibex["Close"].shift(1)).fillna(0.0)

        # Guardamos un diccionario temporal solo para inicialización
        self._temp_ibex_dict = {
            row["Date"].date(): float(row["log_ret_ibex"])
            for _, row in ibex.iterrows()
        }
    # ──────────────────────────────────────────────────────────────────────────
    # Preparación del dataset
    # ──────────────────────────────────────────────────────────────────────────
    def _prepare_columns(self, features):
        excluded_non_numeric = ["Date", "Ticker", "RIC", "Nombre", "Sector", "Industria", "Indice"]
        excluded_price_cols  = ["Open", "High", "Low", "Volume", "Close"]

        if features is not None:
            self.features = list(features)
        else:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            excluded     = [c for c in excluded_non_numeric + excluded_price_cols if c in self.df.columns]
            self.features = [c for c in numeric_cols if c not in excluded]

        if not self.features:
            raise ValueError("No se detectaron features numéricas válidas para el entorno GA-2.")

        missing = [c for c in self.features if c not in self.df.columns]
        if missing:
            raise ValueError(f"Faltan estas features en el dataset: {missing}")

    def _validate_data(self):
        required = {"Date", "Ticker", "Close"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")

        present         = set(self.df["Ticker"].dropna().unique())
        missing_tickers = sorted(set(self.tickers) - present)
        if missing_tickers:
            raise ValueError(f"Tickers ausentes en el dataset: {missing_tickers}")

    def _prepare_dataframe(self):
        for col in ["Close"] + self.features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)

        self.df = self.df.sort_values(["Date", "Ticker"]).copy()
        full_dates = pd.Index(sorted(self.df["Date"].dropna().unique()), name="Date")
        full_index = pd.MultiIndex.from_product(
            [full_dates, self.tickers], names=["Date", "Ticker"]
        )
        self.df = self.df.set_index(["Date", "Ticker"]).reindex(full_index).reset_index()

        for col in ["Close"] + self.features:
            self.df[col] = self.df.groupby("Ticker")[col].ffill().bfill().fillna(0.0)

    def _build_tensor_cache(self):
        self.fechas   = np.array(sorted(self.df["Date"].unique()))
        self.num_days = len(self.fechas)
        
        # --- NUEVO: PRECALCULO DEL IBEX EN ARRAY DE NUMPY ---
        # Mapeamos el diccionario a un array exacto alineado con `self.fechas`
        self.ibex_returns_array = np.array([
            self._temp_ibex_dict.get(pd.Timestamp(f).date(), 0.0) for f in self.fechas
        ], dtype=np.float32)
        # ----------------------------------------------------
        
        if self.num_days <= self.window_size:
            raise ValueError(
                f"No hay suficientes fechas: num_days={self.num_days}, "
                f"window_size={self.window_size}"
            )

        close_pivot = (
            self.df.pivot(index="Date", columns="Ticker", values="Close")
            .reindex(index=self.fechas, columns=self.tickers)
            .ffill().bfill().fillna(0.0)
        )
        self.close_matrix = close_pivot.to_numpy(dtype=np.float32)

        feature_matrices = []
        for feat in self.features:
            pivot = (
                self.df.pivot(index="Date", columns="Ticker", values=feat)
                .reindex(index=self.fechas, columns=self.tickers)
                .ffill().bfill().fillna(0.0)
            )
            feature_matrices.append(pivot.to_numpy(dtype=np.float32))

        self.feature_tensor = np.stack(feature_matrices, axis=-1).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Cartera inicial
    # ──────────────────────────────────────────────────────────────────────────
    def _load_initial_portfolio(self):
        if not self.portfolio_init_path.exists():
            raise FileNotFoundError(
                f"No se encontró la cartera inicial en: {self.portfolio_init_path}"
            )

        p0 = pd.read_csv(self.portfolio_init_path)
        p0.columns = [str(c).strip() for c in p0.columns]
        cols_lower = {c.lower(): c for c in p0.columns}

        # Columna Ticker
        ticker_col = cols_lower.get("ticker")
        # Columna de pesos: acepta Peso_Inicial, Weight, Peso, Weight_Target
        weight_col = (
            cols_lower.get("peso_inicial")
            or cols_lower.get("weight")
            or cols_lower.get("peso")
            or cols_lower.get("weight_target")
        )

        if ticker_col is None or weight_col is None:
            raise ValueError(
                "El CSV de cartera inicial debe tener columnas 'Ticker' y "
                "'Peso_Inicial' (o 'Weight' / 'Peso' / 'Weight_Target')."
            )

        p0 = p0[[ticker_col, weight_col]].copy()
        p0.columns = ["Ticker", "Weight"]
        p0["Ticker"] = p0["Ticker"].astype(str).str.strip()
        p0["Weight"] = pd.to_numeric(p0["Weight"], errors="coerce").fillna(0.0)

        if (p0["Weight"] < 0).any():
            raise ValueError("La cartera inicial no puede contener pesos negativos.")

        agg     = p0.groupby("Ticker", as_index=False)["Weight"].sum()
        unknown = sorted(set(agg["Ticker"]) - set(self.tickers) - {"CASH", "EFECTIVO"})
        if unknown:
            warnings.warn(
        f"Tickers en cartera inicial no presentes en el universo y serán ignorados: {unknown}",
        UserWarning, stacklevel=2
    )
        agg = agg[~agg["Ticker"].isin(unknown)].copy()

        self.initial_target_weights = np.zeros(self.num_assets, dtype=np.float64)
        ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}
        cash_weight   = 0.0

        for _, row in agg.iterrows():
            t, w = row["Ticker"], float(row["Weight"])
            if t in {"CASH", "EFECTIVO"}:
                cash_weight += w
            elif t in ticker_to_idx:
                self.initial_target_weights[ticker_to_idx[t]] += w

        self.initial_target_weights[-1] = cash_weight

        gross = float(self.initial_target_weights.sum())
        if gross <= 0:
            raise ValueError("La cartera inicial no puede estar vacía o sumar 0.")

        # Normalización → suma = 1.0 (garantía de ≤ 100 %)
        self.initial_target_weights /= gross

    # ──────────────────────────────────────────────────────────────────────────
    # Proyección de acciones → pesos válidos (suma ≤ 1, sin apalancamiento)
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _softmax(x):
        z = np.asarray(x, dtype=np.float64) - np.max(x)
        e = np.exp(np.clip(z, -50, 50))
        return e / (np.sum(e) + 1e-12)

    def _project_weights(self, action):
        """
        Convierte los logits del algoritmo genético en pesos válidos:
        1. Softmax → pesos no negativos que suman 1.
        2. Si max_positions está definido, se conservan sólo los top-k activos
           (los demás se llevan a CASH).
        3. Renormalización final → suma = 1.0 siempre.

        La suma de pesos de acciones + CASH = 1 garantiza que la cartera
        nunca supera el 100 % del capital.
        """
        weights = self._softmax(action)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        if self.max_positions is not None:
            k       = (min(self.max_positions, len(weights)))
            top_idx = np.argsort(weights)[-k:]
            mask    = np.zeros_like(weights, dtype=bool)
            mask[top_idx] = True
            weights = np.where(mask, weights, 0.0)

        total = float(weights.sum())
        if total <= 0:
            weights       = np.zeros_like(weights)
            weights[-1]   = 1.0
        else:
            weights /= total

        return weights

    # ──────────────────────────────────────────────────────────────────────────
    # Utilidades de precio y pesos
    # ──────────────────────────────────────────────────────────────────────────
    def _get_prices(self):
        return self.close_matrix[self.current_step].astype(np.float64)

    def _compute_weights(self, prices=None):
        if prices is None:
            prices = self._get_prices()
        pos_values = self.shares_held * prices
        total      = float(np.sum(pos_values) + self.balance)

        if total <= 0:
            w      = np.zeros(self.num_assets, dtype=np.float64)
            w[-1]  = 1.0
            return w

        stock_weights = pos_values / total
        cash_weight   = self.balance / total
        weights       = np.concatenate([stock_weights, [cash_weight]])
        weights       = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights       = np.clip(weights, 0.0, 1.0)
        s             = weights.sum()
        if s <= 0:
            weights[-1] = 1.0
        else:
            weights /= s
        return weights

    # ──────────────────────────────────────────────────────────────────────────
    # Observación
    # ──────────────────────────────────────────────────────────────────────────
    def _get_obs(self):
        feat_window = self.feature_tensor[
            self.current_step - self.window_size : self.current_step
        ]
        feat_window   = np.nan_to_num(feat_window, nan=0.0, posinf=0.0, neginf=0.0)
        feat_window   = np.clip(feat_window, -self.obs_clip, self.obs_clip)
        flat_features = feat_window.reshape(-1)

        prices       = self._get_prices()
        weights      = self._compute_weights(prices)
        drawdown     = self.net_worth / max(self.max_net_worth, 1e-12) - 1.0
        last_ret     = self.daily_return_history[-1]   if self.daily_return_history   else 0.0
        last_turnover = self.turnover_history[-1]       if self.turnover_history       else 0.0
        cash_ratio   = float(weights[-1])

        # Log-retorno del IBEX 35 en el día actual (t) como contexto
        ibex_log_ret_today = float(self.ibex_returns_array[self.current_step])

        portfolio_state = np.concatenate([
            weights,
            np.array(
                [drawdown, last_ret, last_turnover, cash_ratio, ibex_log_ret_today],
                dtype=np.float64,
            ),
        ])

        obs = np.concatenate([flat_features.astype(np.float64), portfolio_state])
        return obs.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Rebalanceo
    # ──────────────────────────────────────────────────────────────────────────
    def _rebalance_to_target(self, target_weights, prices):
        total_equity          = self.balance + np.sum(self.shares_held * prices)
        target_cash           = float(target_weights[-1] * total_equity)
        target_stock_values   = target_weights[:-1] * total_equity
        current_stock_values  = self.shares_held * prices
        current_weights       = self._compute_weights(prices)

        turnover              = float(np.sum(np.abs(target_weights - current_weights)))
        realized_trade_value  = 0.0
        delta_values          = target_stock_values - current_stock_values

        # Ventas primero
        sell_idx = np.where(delta_values < 0)[0]
        for i in sell_idx:
            if prices[i] <= 0:
                continue
            sell_value    = min(-delta_values[i], current_stock_values[i])
            shares_to_sell = np.floor(sell_value / prices[i])
            if shares_to_sell > 0:
                gross = shares_to_sell * prices[i]
                cost  = gross * self.commission
                self.balance           += gross - cost
                self.shares_held[i]    -= shares_to_sell
                realized_trade_value   += gross

        # Compras
        buy_idx = np.where(delta_values > 0)[0]
        for i in buy_idx:
            if prices[i] <= 0:
                continue
            desired_value   = delta_values[i]
            affordable_value = self.balance / (1.0 + self.commission)
            buy_value       = min(desired_value, affordable_value)
            shares_to_buy   = np.floor(buy_value / prices[i])
            if shares_to_buy > 0:
                gross      = shares_to_buy * prices[i]
                cost       = gross * self.commission
                total_cost = gross + cost
                if total_cost <= self.balance:
                    self.balance         -= total_cost
                    self.shares_held[i]  += shares_to_buy
                    realized_trade_value += gross

        # Ajuste de cash si hay déficit
        if self.balance < target_cash:
            deficit = target_cash - self.balance
            order   = np.argsort(self.shares_held * prices)[::-1]
            for i in order:
                if deficit <= 0 or prices[i] <= 0:
                    break
                position_value = self.shares_held[i] * prices[i]
                sell_value     = min(
                    position_value, deficit / max(1.0 - self.commission, 1e-12)
                )
                shares_to_sell = np.floor(sell_value / prices[i])
                if shares_to_sell > 0:
                    gross              = shares_to_sell * prices[i]
                    cost               = gross * self.commission
                    self.balance      += gross - cost
                    self.shares_held[i] -= shares_to_sell
                    realized_trade_value += gross
                    deficit = max(0.0, target_cash - self.balance)

        post_value       = float(self.balance + np.sum(self.shares_held * prices))
        transaction_cost = float(realized_trade_value * self.commission)
        return turnover, transaction_cost, post_value

    # ──────────────────────────────────────────────────────────────────────────
    # Función de recompensa (NÚCLEO DEL CAMBIO RESPECTO A v1)
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_reward(self, prev_net_worth, net_worth):
        """
        Recompensa = log-retorno diario de la cartera − log-retorno diario del IBEX 35.
        Utiliza acceso indexado a self.ibex_returns_array (O(1) ultrarrápido).
        """
        portfolio_log_ret = np.log(net_worth / max(prev_net_worth, 1e-12))
        
        # Acceso instantáneo por el índice del paso actual
        ibex_log_ret = float(self.ibex_returns_array[self.current_step])

        reward = float(portfolio_log_ret - ibex_log_ret)
        return reward, float(portfolio_log_ret), ibex_log_ret

    # ──────────────────────────────────────────────────────────────────────────
    # Info de paso
    # ──────────────────────────────────────────────────────────────────────────
    def _get_info(self, turnover, transaction_cost, daily_return,
                  ibex_log_ret=0.0, reward=0.0):
        prices   = self._get_prices()
        weights  = self._compute_weights(prices)
        cardinality = int(np.sum(weights[:-1] > 1e-12))
        info = {
            "date"             : pd.Timestamp(self.fechas[self.current_step]),
            "step"             : int(self.current_step),
            "prices"           : prices.copy(),
            "shares_held"      : self.shares_held.copy(),
            "balance"          : float(self.balance),
            "net_worth"        : float(self.net_worth),
            "weights"          : weights.copy(),
            "cash_weight"      : float(weights[-1]),
            "turnover"         : float(turnover),
            "transaction_cost" : float(transaction_cost),
            "daily_return"     : float(daily_return),
            "ibex_log_ret"     : float(ibex_log_ret),
            "reward"           : float(reward),
            "num_active_positions": cardinality,
            "drawdown"         : float(self.net_worth / max(self.max_net_worth, 1e-12) - 1.0),
        }
        self.last_info = info
        return info

    # ──────────────────────────────────────────────────────────────────────────
    # Aplicación de la cartera inicial
    # ──────────────────────────────────────────────────────────────────────────
    def _apply_initial_portfolio(self):
        """
        Inicializa la cartera 100% en liquidez (CASH).
        Elimina la necesidad de leer el archivo CSV y simular compras iniciales.
        """
        self.balance     = self.initial_balance
        self.shares_held = np.zeros(self.num_stocks, dtype=np.float64)

        self.net_worth      = float(self.balance)
        self.prev_net_worth = self.net_worth
        self.max_net_worth  = self.net_worth

    # ──────────────────────────────────────────────────────────────────────────
    # reset / step / render
    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step            = self.window_size
        self.turnover_history        = []
        self.transaction_cost_history = []
        self.action_history          = []
        self.weight_history          = []
        self.date_history            = []
        self.net_worth_history       = []
        self.daily_return_history    = []
        self.ibex_log_ret_history    = []
        self.reward_history          = []

        self._apply_initial_portfolio()

        self.net_worth_history.append(self.net_worth)
        self.date_history.append(pd.Timestamp(self.fechas[self.current_step]))
        self.weight_history.append(self._compute_weights(self._get_prices()).copy())

        obs  = self._get_obs()
        info = self._get_info(
            turnover=0.0, transaction_cost=0.0, daily_return=0.0,
            ibex_log_ret=0.0, reward=0.0,
        )
        return obs, info

    def step(self, action):
        prices = self._get_prices()
        action = np.asarray(action, dtype=np.float64).flatten()
        if len(action) != self.num_assets:
            raise ValueError(
                f"La acción debe tener longitud {self.num_assets}, recibida {len(action)}"
            )

        # ── Proyección de pesos (garantía de ≤ 100 %) ────────────────────────
        target_weights = self._project_weights(action)

        # ── Rebalanceo ────────────────────────────────────────────────────────
        turnover, transaction_cost = 0.0, 0.0
        if (self.current_step - self.window_size) % self.rebalance_freq == 0:
            turnover, transaction_cost, _ = self._rebalance_to_target(
                target_weights, prices
            )

        # ── Avance temporal ───────────────────────────────────────────────────
        self.current_step += 1
        terminated = self.current_step >= len(self.fechas) - 1
        truncated  = self.net_worth <= self.initial_balance * UMBRAL_QUIEBRA

        # ── Nuevos precios y valor de cartera ─────────────────────────────────
        new_prices          = self._get_prices()
        self.prev_net_worth = self.net_worth
        self.net_worth      = float(self.balance + np.sum(self.shares_held * new_prices))
        self.max_net_worth  = max(self.max_net_worth, self.net_worth)

        # ── Retorno diario (aritmético, para historial) ───────────────────────
        daily_return = self.net_worth / max(self.prev_net_worth, 1e-12) - 1.0

        # ── Recompensa: diferencial log-retorno cartera − IBEX 35 ─────────────
        current_date = self.fechas[self.current_step]
        reward, portfolio_log_ret, ibex_log_ret = self._compute_reward(
            prev_net_worth=self.prev_net_worth,
            net_worth=self.net_worth,
        )

        # Penalización por bancarrota
        if truncated:
            reward    += CASTIGO_BANCARROTA
            terminated = True

        # ── Registros históricos ──────────────────────────────────────────────
        self.net_worth_history.append(self.net_worth)
        self.daily_return_history.append(float(daily_return))
        self.ibex_log_ret_history.append(float(ibex_log_ret))
        self.reward_history.append(float(reward))
        self.turnover_history.append(float(turnover))
        self.transaction_cost_history.append(float(transaction_cost))
        self.action_history.append(target_weights.copy())
        self.weight_history.append(self._compute_weights(new_prices).copy())
        self.date_history.append(pd.Timestamp(current_date))

        obs  = self._get_obs()
        info = self._get_info(
            turnover=turnover,
            transaction_cost=transaction_cost,
            daily_return=daily_return,
            ibex_log_ret=ibex_log_ret,
            reward=reward,
        )
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        print(
            f"Step {self.current_step} | "
            f"Date {self.fechas[self.current_step]} | "
            f"Net Worth {self.net_worth:,.2f} € | "
            f"Balance {self.balance:,.2f} € | "
            f"Reward {self.reward_history[-1] if self.reward_history else 0.0:+.6f}"
        )
