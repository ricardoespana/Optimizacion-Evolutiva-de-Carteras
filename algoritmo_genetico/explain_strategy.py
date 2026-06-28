"""
================================================================================
EXPLAINABLE AI (XAI) - DESCIFRADO DE ESTRATEGIA DE ALGORITMO GENETICO CON SHAP
================================================================================
Script de análisis de interpretabilidad para un Algoritmo Genético de gestión
de carteras. Utiliza un Modelo Sustituto (Random Forest) para aproximar la
función de decisión del GA y luego aplica SHAP (SHapley Additive exPlanations)
para cuantificar la contribución marginal de cada variable explicativa.

Flujo del análisis:
    1. Carga de features de mercado y decisiones históricas del GA.
    2. Cruce de decisiones con indicadores técnicos por (Fecha, Ticker).
    3. Entrenamiento de un Random Forest como modelo sustituto.
    4. Validación estadística del modelo sustituto (métricas de bondad de ajuste).
    5. Cálculo de valores SHAP sobre una muestra representativa.
    6. Exportación de gráficos y reporte de métricas.

Autor: Ricardo España Lopez
Fecha: Mayo 2026
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import sys
import json
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

print("=" * 70)
print("INICIANDO ANALISIS DE ESTRATEGIA DEL ALGORITMO GENETICO (XAI - SHAP)")
print("=" * 70)
print(f"Fecha de ejecucion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()


# ==============================================================================
# 1. CONFIGURACION DEL EXPERIMENTO
# ==============================================================================
# El identificador del experimento determina las rutas de entrada y salida.
# Modificar ID_EXPERIMENTO para apuntar a otro experimento sin tocar el resto.

ID_EXPERIMENTO = "GA_MultiStock_v29"

RUTA_EXPERIMENTO  = f"GA_Env/experimentos/{ID_EXPERIMENTO}"
RUTA_BACKTEST     = f"{RUTA_EXPERIMENTO}/backtest_results"
RUTA_DATOS_MERCADO = "GA_Env/data/dataset_GA_estandarizado_tecnicos.csv"
RUTA_COMPOSICION  = f"{RUTA_BACKTEST}/portfolio_composition_history.csv"
DIR_SALIDA        = f"{RUTA_EXPERIMENTO}/explicabilidad_shap"

os.makedirs(DIR_SALIDA, exist_ok=True)

# Columnas que no constituyen variables explicativas del modelo:
# fechas, identificadores de texto, precio de cierre en crudo,
# rentabilidad (variable a predecir internamente por el entorno),
# y el patrimonio neto (variable de estado, no de señal).
COLUMNAS_EXCLUIR = [
    'Date', 'Ticker', 'Close',
    'Rentabilidad_Diaria', 'Net_Worth'
]

# Parámetros del Random Forest sustituto
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH    = 10
RF_RANDOM_STATE = 42
RF_N_JOBS       = -1

# Tamaño máximo de muestra para el cálculo SHAP (limitado por coste computacional)
SHAP_MAX_SAMPLE = 2000

# Numero de folds para validacion cruzada
CV_N_FOLDS = 5


# ==============================================================================
# 2. CARGA DE DATOS DE MERCADO (FEATURES)
# ==============================================================================
# El archivo de mercado contiene, por (Fecha, Ticker), todos los indicadores
# técnicos y de precio que el entorno presentaba al agente como observación.

print("[ BLOQUE 2 ] Cargando datos de mercado (features)...")

if not os.path.exists(RUTA_DATOS_MERCADO):
    print(f"  ERROR: No se encuentra el archivo de datos: {RUTA_DATOS_MERCADO}")
    sys.exit(1)

df_datos = pd.read_csv(RUTA_DATOS_MERCADO)
df_datos['Date'] = pd.to_datetime(df_datos['Date'])

# Seleccionamos únicamente columnas numéricas que no están en la lista de exclusión.
# Esto garantiza que cualquier feature nueva que se añada al dataset sea incluida
# automáticamente sin necesidad de modificar el script.
features_disponibles = [
    col for col in df_datos.columns
    if col not in COLUMNAS_EXCLUIR
    and df_datos[col].dtype in ['float64', 'int64']
]

print(f"  -> Registros cargados  : {len(df_datos):,}")
print(f"  -> Variables detectadas: {len(features_disponibles)}")
print(f"  -> Primeras variables  : {features_disponibles[:8]}")
print()


# ==============================================================================
# 3. EXTRACCION DE DECISIONES DEL ALGORITMO GENETICO
# ==============================================================================
# El GA genera un archivo con la composición histórica de la cartera en formato
# ancho (Wide): una fila por fecha y una columna por ticker.
# Transformamos este formato a largo (Long): una fila por (Fecha, Ticker, Peso).

print("[ BLOQUE 3 ] Extrayendo decisiones del Algoritmo Genetico...")

if not os.path.exists(RUTA_COMPOSICION):
    print(f"  ERROR: No se encuentra el archivo de composicion: {RUTA_COMPOSICION}")
    print(f"  Ruta esperada: {RUTA_COMPOSICION}")
    sys.exit(1)

df_wide = pd.read_csv(RUTA_COMPOSICION)
df_wide['Date'] = pd.to_datetime(df_wide['Date'])

# Pivotamos de formato ancho a largo para homogeneizar con el dataset de mercado
df_decisiones = df_wide.melt(
    id_vars=['Date'],
    var_name='Ticker',
    value_name='Peso_Asignado'
)

# Eliminamos posiciones nulas y la asignación a CASH.
# CASH no dispone de indicadores técnicos, por lo que no puede ser analizado
# por el modelo sustituto (no tiene contraparte en el dataset de mercado).
n_antes = len(df_decisiones)
df_decisiones = df_decisiones[
    (df_decisiones['Peso_Asignado'] > 0) &
    (df_decisiones['Ticker'] != 'CASH')
].copy()
n_despues = len(df_decisiones)

print(f"  -> Fechas en el historico  : {df_wide['Date'].nunique():,}")
print(f"  -> Tickers identificados   : {df_wide.shape[1] - 1}")
print(f"  -> Decisiones totales      : {n_antes:,}")
print(f"  -> Decisiones con peso > 0 : {n_despues:,}  (excluidos CASH y pesos nulos)")
print()


# ==============================================================================
# 4. CONSTRUCCION DEL DATASET ANALITICO
# ==============================================================================
# Cruzamos las decisiones del GA con los indicadores técnicos correspondientes
# a la misma (Fecha, Ticker). El resultado es el dataset de entrenamiento del
# modelo sustituto: X = indicadores, y = peso asignado por el GA.

print("[ BLOQUE 4 ] Construyendo dataset analitico (decisiones + features)...")

df_analisis = pd.merge(
    df_decisiones,
    df_datos,
    on=['Date', 'Ticker'],
    how='inner'
)

# Advertencia si el cruce elimina muchas observaciones (posible desalineación de fechas)
n_perdidos = n_despues - len(df_analisis)
if n_perdidos > 0:
    porcentaje_perdido = (n_perdidos / n_despues) * 100
    print(f"  AVISO: {n_perdidos:,} decisiones ({porcentaje_perdido:.1f}%) no tienen "
          f"contraparte en el dataset de mercado.")
    print(f"  Verifique la alineacion de fechas entre ambos archivos.")

print(f"  -> Observaciones para entrenamiento: {len(df_analisis):,}")
print(f"  -> Rango de fechas: {df_analisis['Date'].min().date()} a {df_analisis['Date'].max().date()}")
print(f"  -> Tickers en el analisis: {df_analisis['Ticker'].nunique()}")
print()

X = df_analisis[features_disponibles].fillna(0)
y = df_analisis['Peso_Asignado']


# ==============================================================================
# 5. ENTRENAMIENTO DEL MODELO SUSTITUTO
# ==============================================================================
# Entrenamos un Random Forest sobre el dataset completo. El objetivo es replicar
# (clonar) la lógica del GA: dado un vector de indicadores de mercado, predecir
# qué peso asignaría el GA a ese activo.
# Un R2 alto indica que el sustituto captura bien la lógica del GA.

print("[ BLOQUE 5 ] Entrenando Modelo Sustituto (Random Forest)...")

modelo_sustituto = RandomForestRegressor(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    random_state=RF_RANDOM_STATE,
    n_jobs=RF_N_JOBS
)
modelo_sustituto.fit(X, y)

y_pred_full = modelo_sustituto.predict(X)

r2_train    = r2_score(y, y_pred_full)
rmse_train  = np.sqrt(mean_squared_error(y, y_pred_full))
mae_train   = mean_absolute_error(y, y_pred_full)

print(f"  -> R2   (train, ajuste in-sample): {r2_train:.4f}")
print(f"  -> RMSE (train):                   {rmse_train:.6f}")
print(f"  -> MAE  (train):                   {mae_train:.6f}")

if r2_train < 0.50:
    print("  AVISO: R2 < 0.50. El sustituto captura menos de la mitad de la varianza.")
    print("         Los valores SHAP deben interpretarse con precaucion.")
print()


# ==============================================================================
# 6. VALIDACION ESTADISTICA DEL MODELO SUSTITUTO
# ==============================================================================
# Evaluamos el modelo sustituto con validación cruzada k-fold para obtener
# estimaciones no sesgadas del rendimiento. Calculamos múltiples métricas de
# bondad de ajuste y las exportamos a un archivo de texto.
#
# Métricas calculadas:
#   - R2           : Coeficiente de determinación (varianza explicada).
#   - RMSE         : Raíz del Error Cuadrático Medio (misma escala que el peso).
#   - MAE          : Error Absoluto Medio (robustez frente a outliers).
#   - MAPE         : Error Porcentual Absoluto Medio (interpretación relativa).
#   - Correlacion  : Correlación de Pearson entre predicciones y valores reales.
#   - Bias         : Sesgo promedio del modelo (tendencia sistemática a sobre/infravalorar).

print(f"[ BLOQUE 6 ] Ejecutando validacion cruzada ({CV_N_FOLDS}-fold)...")

kf = KFold(n_splits=CV_N_FOLDS, shuffle=True, random_state=RF_RANDOM_STATE)

cv_r2   = cross_val_score(modelo_sustituto, X, y, cv=kf, scoring='r2',
                           n_jobs=RF_N_JOBS)
cv_rmse = cross_val_score(modelo_sustituto, X, y, cv=kf,
                           scoring='neg_root_mean_squared_error',
                           n_jobs=RF_N_JOBS)
cv_mae  = cross_val_score(modelo_sustituto, X, y, cv=kf,
                           scoring='neg_mean_absolute_error',
                           n_jobs=RF_N_JOBS)

cv_rmse_pos = np.abs(cv_rmse)
cv_mae_pos  = np.abs(cv_mae)

# Calculamos correlacion y sesgo a partir del conjunto completo de predicciones
correlacion_pearson = np.corrcoef(y, y_pred_full)[0, 1]
bias_promedio       = np.mean(y_pred_full - y)

# MAPE: evitamos division por cero excluyendo pesos nulos
mask_nonzero = y > 0
mape = np.mean(np.abs((y[mask_nonzero] - y_pred_full[mask_nonzero]) / y[mask_nonzero])) * 100

print(f"  -> R2 CV  (media +/- std): {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
print(f"  -> RMSE CV(media +/- std): {cv_rmse_pos.mean():.6f} +/- {cv_rmse_pos.std():.6f}")
print(f"  -> MAE CV (media +/- std): {cv_mae_pos.mean():.6f} +/- {cv_mae_pos.std():.6f}")
print(f"  -> MAPE (full sample)    : {mape:.2f}%")
print(f"  -> Correlacion de Pearson: {correlacion_pearson:.4f}")
print(f"  -> Bias promedio         : {bias_promedio:.6f}")
print()


# --- Exportacion del reporte de metricas ---
ruta_metricas = os.path.join(DIR_SALIDA, f"metricas_validacion_GA_{ID_EXPERIMENTO}.txt")

with open(ruta_metricas, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("REPORTE DE VALIDACION - MODELO SUSTITUTO (RANDOM FOREST)\n")
    f.write(f"Experimento : {ID_EXPERIMENTO}\n")
    f.write(f"Generado    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")

    f.write("CONFIGURACION DEL MODELO SUSTITUTO\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Algoritmo           : Random Forest Regressor\n")
    f.write(f"  Numero de arboles   : {RF_N_ESTIMATORS}\n")
    f.write(f"  Profundidad maxima  : {RF_MAX_DEPTH}\n")
    f.write(f"  Semilla aleatoria   : {RF_RANDOM_STATE}\n")
    f.write(f"  Variables utilizadas: {len(features_disponibles)}\n")
    f.write(f"  Observaciones (n)   : {len(X):,}\n\n")

    f.write("DATASET ANALITICO\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Rango de fechas     : {df_analisis['Date'].min().date()} a {df_analisis['Date'].max().date()}\n")
    f.write(f"  Tickers analizados  : {df_analisis['Ticker'].nunique()}\n")
    f.write(f"  Peso asignado medio : {y.mean():.4f}\n")
    f.write(f"  Desv. estandar peso : {y.std():.4f}\n")
    f.write(f"  Peso minimo         : {y.min():.4f}\n")
    f.write(f"  Peso maximo         : {y.max():.4f}\n\n")

    f.write("METRICAS DE BONDAD DE AJUSTE (IN-SAMPLE)\n")
    f.write("-" * 40 + "\n")
    f.write(f"  R2  (coef. determinacion) : {r2_train:.4f}\n")
    f.write(f"  RMSE                      : {rmse_train:.6f}\n")
    f.write(f"  MAE                       : {mae_train:.6f}\n")
    f.write(f"  MAPE                      : {mape:.2f}%\n")
    f.write(f"  Correlacion de Pearson    : {correlacion_pearson:.4f}\n")
    f.write(f"  Bias promedio             : {bias_promedio:.6f}\n\n")

    f.write(f"METRICAS DE VALIDACION CRUZADA ({CV_N_FOLDS}-FOLD)\n")
    f.write("-" * 40 + "\n")
    f.write(f"  R2   - Media : {cv_r2.mean():.4f}   | Std : {cv_r2.std():.4f}\n")
    f.write(f"  R2   - Min   : {cv_r2.min():.4f}   | Max : {cv_r2.max():.4f}\n")
    f.write(f"  R2   - por fold: {', '.join([f'{v:.4f}' for v in cv_r2])}\n\n")
    f.write(f"  RMSE - Media : {cv_rmse_pos.mean():.6f} | Std : {cv_rmse_pos.std():.6f}\n")
    f.write(f"  RMSE - Min   : {cv_rmse_pos.min():.6f} | Max : {cv_rmse_pos.max():.6f}\n")
    f.write(f"  RMSE - por fold: {', '.join([f'{v:.6f}' for v in cv_rmse_pos])}\n\n")
    f.write(f"  MAE  - Media : {cv_mae_pos.mean():.6f} | Std : {cv_mae_pos.std():.6f}\n")
    f.write(f"  MAE  - Min   : {cv_mae_pos.min():.6f} | Max : {cv_mae_pos.max():.6f}\n")
    f.write(f"  MAE  - por fold: {', '.join([f'{v:.6f}' for v in cv_mae_pos])}\n\n")

    f.write("INTERPRETACION DE LOS RESULTADOS\n")
    f.write("-" * 40 + "\n")
    if cv_r2.mean() >= 0.80:
        nivel_ajuste = "EXCELENTE (R2 >= 0.80): El sustituto replica fielmente la logica del GA."
    elif cv_r2.mean() >= 0.60:
        nivel_ajuste = "BUENO (R2 entre 0.60 y 0.80): Los valores SHAP son confiables."
    elif cv_r2.mean() >= 0.40:
        nivel_ajuste = "MODERADO (R2 entre 0.40 y 0.60): Interpretar los valores SHAP con cautela."
    else:
        nivel_ajuste = "BAJO (R2 < 0.40): Alta incertidumbre. Considerar ampliar features o ajustar RF."

    f.write(f"  Nivel de ajuste: {nivel_ajuste}\n\n")

    overfitting_gap = r2_train - cv_r2.mean()
    if overfitting_gap > 0.15:
        f.write(f"  AVISO: Posible sobreajuste detectado.\n")
        f.write(f"         R2 in-sample ({r2_train:.4f}) supera en {overfitting_gap:.4f} al R2 CV ({cv_r2.mean():.4f}).\n")
        f.write(f"         Considerar reducir max_depth o aumentar min_samples_leaf.\n")
    else:
        f.write(f"  Sin indicios de sobreajuste significativo (gap R2 = {overfitting_gap:.4f}).\n")

    f.write("\n" + "=" * 70 + "\n")

print(f"  -> Reporte de metricas exportado: {ruta_metricas}")
print()


# ==============================================================================
# 7. CALCULO DE VALORES SHAP
# ==============================================================================
# SHAP (SHapley Additive exPlanations) descompone cada predicción del modelo
# sustituto en contribuciones aditivas de cada variable. Permite responder:
# "¿Qué indicadores llevaron al GA a asignar más o menos peso a este activo?"
#
# TreeExplainer es el método exacto para árboles de decisión y Random Forests.
# Es determinístico y no requiere muestreo de las perturbaciones.

print("[ BLOQUE 7 ] Calculando valores SHAP...")

explainer = shap.TreeExplainer(modelo_sustituto)

# Tomamos una muestra estratificada si el dataset es grande para reducir
# el coste computacional del cálculo de valores SHAP sin perder representatividad.
if len(X) > SHAP_MAX_SAMPLE:
    X_muestra = X.sample(SHAP_MAX_SAMPLE, random_state=RF_RANDOM_STATE)
    print(f"  -> Dataset grande ({len(X):,} obs.). Usando muestra de {SHAP_MAX_SAMPLE:,} obs.")
else:
    X_muestra = X.copy()
    print(f"  -> Calculando SHAP sobre el dataset completo ({len(X):,} obs.).")

shap_values = explainer.shap_values(X_muestra)
print(f"  -> Valores SHAP calculados correctamente.")
print()


# ==============================================================================
# 8. GENERACION DE VISUALIZACIONES SHAP
# ==============================================================================
# Generamos dos tipos de gráficos complementarios:
#
# 8a. Summary Plot (Beeswarm): Muestra la distribución de los valores SHAP de
#     cada variable. El color indica el valor de la feature (rojo=alto, azul=bajo).
#     Permite detectar tanto la importancia como la dirección del efecto.
#
# 8b. Bar Plot (Importancia absoluta): Muestra el valor SHAP promedio en valor
#     absoluto. Es el ranking de importancia agregada de cada variable.

print("[ BLOQUE 8 ] Generando visualizaciones SHAP...")

# --- 8a. Summary Plot (Beeswarm) ---
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_muestra,
    show=False,
    cmap='coolwarm'
)
plt.title(
    f'Estrategia de Asignacion del Algoritmo Genetico - SHAP Summary\n'
    f'Experimento: {ID_EXPERIMENTO}',
    fontsize=13,
    fontweight='bold',
    pad=12
)
plt.tight_layout()
ruta_summary = os.path.join(DIR_SALIDA, f"shap_summary_GA_{ID_EXPERIMENTO}.png")
plt.savefig(ruta_summary, dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> Summary plot guardado en: {ruta_summary}")

# --- 8b. Bar Plot (Importancia absoluta) ---
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_muestra,
    plot_type="bar",
    show=False,
    color='steelblue'
)
plt.title(
    f'Importancia de Variables (SHAP Absoluto) - Algoritmo Genetico\n'
    f'Experimento: {ID_EXPERIMENTO}',
    fontsize=13,
    fontweight='bold',
    pad=12
)
plt.tight_layout()
ruta_barras = os.path.join(DIR_SALIDA, f"shap_importance_bar_GA_{ID_EXPERIMENTO}.png")
plt.savefig(ruta_barras, dpi=300, bbox_inches='tight')
plt.close()
print(f"  -> Bar plot guardado en   : {ruta_barras}")
print()


# ==============================================================================
# 9. RESUMEN DE VARIABLES MAS IMPORTANTES (TEXTO)
# ==============================================================================
# Calculamos el ranking de importancia media absoluta de SHAP y lo añadimos
# al reporte de métricas para facilitar la interpretación textual del análisis.

importancia_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=features_disponibles
).sort_values(ascending=False)

with open(ruta_metricas, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("RANKING DE VARIABLES POR IMPORTANCIA SHAP (|SHAP| MEDIO)\n")
    f.write("-" * 40 + "\n")
    for rk, (var, val) in enumerate(importancia_shap.items(), start=1):
        f.write(f"  {rk:>3}. {var:<35} |SHAP| medio = {val:.6f}\n")
    f.write("\n" + "=" * 70 + "\n")

print(f"[ BLOQUE 9 ] Ranking SHAP anadido al reporte de metricas.")
print()
print("  Top 10 variables mas influyentes:")
for rk, (var, val) in enumerate(importancia_shap.head(10).items(), start=1):
    print(f"  {rk:>3}. {var:<35} |SHAP| = {val:.6f}")
print()


# ==============================================================================
# 10. RESUMEN FINAL DE EJECUCION
# ==============================================================================
print("=" * 70)
print("ANALISIS SHAP DEL ALGORITMO GENETICO COMPLETADO")
print("=" * 70)
print(f"Experimento analizado : {ID_EXPERIMENTO}")
print(f"Observaciones         : {len(df_analisis):,}")
print(f"Variables analizadas  : {len(features_disponibles)}")
print(f"R2 CV (media)         : {cv_r2.mean():.4f}")
print()
print("Archivos generados:")
print(f"  [METRICAS]  {ruta_metricas}")
print(f"  [GRAFICO]   {ruta_summary}")
print(f"  [GRAFICO]   {ruta_barras}")
print("=" * 70)
