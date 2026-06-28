#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import random
import logging
import json

"""
EL SCRIPT BUSCA ALEATORIAMENTE 80 EMPRESAS Y PREPARA EL DATASET PARA PODER HACER LA
COMPARATIVA GENERALIZADA DEL MODELO ENTRENADO CON EL IBEX
"""
# Configuración básica de logging para monitorizar la ejecución
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# FASE 1: GENERADOR DEL DATASET COMPLETO (CON NLP SIMULADO)
# ==============================================================================
def generar_dataset_completo(archivo_cotizaciones, archivo_fundamentales, ruta_salida_completo):
    print("\n" + "="*70)
    print("FASE 1: GENERANDO DATASET COMPLETO S&P 500 (CON NLP SIMULADO)")
    print("="*70)

    # ---------------------------------------------------------
    # PASO 1.1: Carga de Cotizaciones y Cálculos Dinámicos
    # ---------------------------------------------------------
    logging.info("Cargando Cotizaciones y calculando Rentabilidad/Volatilidad...")
    if not os.path.exists(archivo_cotizaciones):
        raise FileNotFoundError(f"No se encuentra el archivo: {archivo_cotizaciones}")
        
    df_base = pd.read_csv(archivo_cotizaciones)
    df_base['Ticker'] = df_base['Ticker'].astype(str).str.strip()
    df_base['Date'] = pd.to_datetime(df_base['Date'], utc=True).dt.tz_localize(None).dt.normalize()
    df_base = df_base.sort_values(['Ticker', 'Date'])

    rentabilidad = df_base.groupby('Ticker')['Close'].pct_change()
    volatilidad = df_base.groupby('Ticker')['Close'].transform(lambda x: x.pct_change().rolling(window=20).std())

    if 'Volume' in df_base.columns:
        idx_volume = df_base.columns.get_loc('Volume')
        df_base.insert(idx_volume + 1, 'Rentabilidad_Diaria', rentabilidad)
        df_base.insert(idx_volume + 2, 'Volatilidad_Diaria', volatilidad)

    print(f"  [CHECK 1.1] Cotizaciones cargadas: {len(df_base)} filas, {df_base['Ticker'].nunique()} tickers únicos.")

    # ---------------------------------------------------------
    # PASO 1.2: Simulación de Sentimiento NLP (-1, 0, 1)
    # ---------------------------------------------------------
    logging.info("Simulando datos NLP (Sentimiento_Num)...")
    # Generamos probabilidades realistas: 60% neutral, 20% positivo, 20% negativo
    df_base['Sentimiento_Num'] = np.random.choice([-1, 0, 1], size=len(df_base), p=[0.2, 0.6, 0.2])
    
    print(f"  [CHECK 1.2] Sentimiento simulado con valores: {df_base['Sentimiento_Num'].unique()}")

    # ---------------------------------------------------------
    # PASO 1.3: Procesamiento de Fundamentales
    # ---------------------------------------------------------
    logging.info("Cargando Fundamentales y alineando fechas...")
    if not os.path.exists(archivo_fundamentales):
        raise FileNotFoundError(f"No se encuentra el archivo: {archivo_fundamentales}")
        
    df_fund = pd.read_csv(archivo_fundamentales)
    df_fund['Ticker'] = df_fund['Ticker'].astype(str).str.strip()
    df_fund['Date'] = pd.to_datetime(df_fund['Date'], errors='coerce').dt.normalize()
    df_fund = df_fund.dropna(subset=['Date']).sort_values(['Ticker', 'Date']).reset_index(drop=True)

    lista_fund = []
    tickers_universo = df_base['Ticker'].dropna().unique()

    for ticker in tickers_universo:
        base_ticker = df_base.loc[df_base['Ticker'] == ticker, ['Ticker', 'Date']].sort_values('Date').reset_index(drop=True)
        fund_ticker = df_fund.loc[df_fund['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)

        if fund_ticker.empty:
            continue

        alineado = pd.merge_asof(base_ticker, fund_ticker, on='Date', direction='backward')
        alineado['Ticker'] = ticker
        lista_fund.append(alineado)

    df_fund_final = pd.concat(lista_fund, ignore_index=True)
    print(f"  [CHECK 1.3] Fundamentales alineados: {len(df_fund_final)} filas procesadas.")

    # ---------------------------------------------------------
    # PASO 1.4: Merge y Dataset Crudo
    # ---------------------------------------------------------
    logging.info("Generando Dataset Crudo (Merge total)...")
    df_crudo = pd.merge(df_base, df_fund_final, on=['Ticker', 'Date'], how='left')

    # Limpieza inmediata de columnas duplicadas generadas por el merge (ej. Ticker_y, Date_y)
    columnas_duplicadas = [c for c in df_crudo.columns if c.endswith('_y') or c.endswith('_x') or c.endswith('_fund')]
    if columnas_duplicadas:
        df_crudo.drop(columns=columnas_duplicadas, inplace=True)
        # Renombrar posibles '_x' si quedaron
        df_crudo.columns = df_crudo.columns.str.replace('_x', '', regex=False)

    descriptivas_a_excluir = ['Ticker', 'Date', 'Instrument', 'instrument', 'Nombre', 'Indice',
                              'TRBC_Sector', 'Grupo_Contable', 'Sector', 'Industria', 'Nombre_fund', 'Indice_fund']
    
    # Recalculamos columnas operativas de forma segura
    cols_fundamentales = [c for c in df_fund_final.columns if c not in descriptivas_a_excluir and c in df_crudo.columns]

    df_crudo[cols_fundamentales] = df_crudo.groupby('Ticker')[cols_fundamentales].ffill()
    
    cols_instrument = [c for c in ['instrument', 'Instrument'] if c in df_crudo.columns]
    if cols_instrument:
        df_crudo.drop(columns=cols_instrument, inplace=True)

    df_crudo = df_crudo[df_crudo['Date'] >= '2021-01-01'].copy()
    print(f"  [CHECK 1.4] Dataset Crudo generado desde 2021: {len(df_crudo)} filas.")

    # ---------------------------------------------------------
    # PASO 1.5: Ajustes de Entrenamiento (Decaimiento y NaNs)
    # ---------------------------------------------------------
    logging.info("Aplicando limpieza final y decaimiento exponencial al sentimiento...")
    df_train = df_crudo.copy()

    vars_a_borrar = ['SCORE_FINAL', 'Anio', 'Trimestre', 'RIC_fund', 'Nombre_fund', 
                     'Indice_fund', 'TRBC_Sector', 'Grupo_Contable', 'Net Interest Income', 
                     'Fecha', 'Depreciation And Amortization', 'Cash', 
                     'Capital Expenditures, Cumulative', 'Sector_fund', 'Industria_fund', 'TRBC_Industry']
    
    # Borrado seguro
    df_train.drop(columns=[c for c in vars_a_borrar if c in df_train.columns], inplace=True)

    df_train = df_train.sort_values(['Ticker', 'Date'])
    
    # [CORRECCIÓN CRÍTICA] Recalculamos las columnas seguras que REALMENTE están en el dataset ahora mismo
    cols_fundamentales_seguras = [c for c in cols_fundamentales if c in df_train.columns and c not in ['Ticker', 'Date']]
    
    if cols_fundamentales_seguras:
        df_train[cols_fundamentales_seguras] = df_train.groupby('Ticker')[cols_fundamentales_seguras].ffill()

    # Tratamiento de Sentimiento NLP Simulado (Decaimiento exponencial)
    if 'Sentimiento_Num' in df_train.columns:
        df_train['Sentimiento_Num'] = df_train.groupby('Ticker')['Sentimiento_Num'].ffill()
        df_train['Sentimiento_Num'] = df_train['Sentimiento_Num'].fillna(0)
        df_train['Sentimiento_Num'] = df_train.groupby('Ticker')['Sentimiento_Num'].transform(lambda x: x.ewm(span=5, adjust=False).mean())

    # Agrupación segura de indicadores técnicos (verificando existencia)
    cols_no_tecnicas = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Rentabilidad_Diaria', 'Volatilidad_Diaria', 'Sentimiento_Num'] + cols_fundamentales_seguras
    cols_tecnicas = [c for c in df_train.columns if c not in cols_no_tecnicas]

    if cols_tecnicas:
        df_train[cols_tecnicas] = df_train[cols_tecnicas].fillna(0)
        
    df_train = df_train.dropna(subset=['Volatilidad_Diaria'])
    df_train.fillna(0, inplace=True)

    df_train.to_csv(ruta_salida_completo, index=False)
    print(f"  [CHECK 1.5] Fase 1 Completada. Dataset guardado en: {ruta_salida_completo}")
    return ruta_salida_completo


# ==============================================================================
# FASE 2: ADAPTACIÓN ZERO-SHOT Y VALIDACIÓN (BACKTEST)
# ==============================================================================
def preparar_y_validar_sp500(ruta_entrada: str, ruta_salida: str, ruta_json: str, semilla: int = 42):
    print("\n" + "="*70)
    print("FASE 2: ADAPTACIÓN Y ESTANDARIZACIÓN PARA EL MODELO GA (ZERO-SHOT)")
    print("="*70)

    try:
        # ---------------------------------------------------------
        # PASO 2.1: Lectura del JSON del Modelo
        # ---------------------------------------------------------
        if not os.path.exists(ruta_json):
            raise FileNotFoundError(f"No se encuentra la configuración: {ruta_json}")
            
        with open(ruta_json, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        features_req = config.get("FEATURES_DETECTADAS", [])
        empresas_req = len(config.get("TICKERS_ELEGIBLES", []))
        
        print(f"  [CHECK 2.1] El JSON exige: {empresas_req} empresas y {len(features_req)} features.")

        # ---------------------------------------------------------
        # PASO 2.2: Extracción de la Muestra
        # ---------------------------------------------------------
        df = pd.read_csv(ruta_entrada)
        empresas_unicas = df['Ticker'].unique()

        if empresas_req > len(empresas_unicas):
            raise ValueError(f"Faltan empresas. El modelo pide {empresas_req}, pero solo hay {len(empresas_unicas)}.")

        random.seed(semilla)
        tickers_seleccionados = random.sample(list(empresas_unicas), k=empresas_req)
        df_muestra = df[df['Ticker'].isin(tickers_seleccionados)].copy()
        
        print(f"  [CHECK 2.2] Extraídas {df_muestra['Ticker'].nunique()} empresas aleatorias. Filas: {len(df_muestra)}.")

        # ---------------------------------------------------------
        # PASO 2.3: Estandarización Fija (2021-2023)
        # ---------------------------------------------------------
        df_muestra['Date'] = pd.to_datetime(df_muestra['Date'])
        mask_train = (df_muestra['Date'].dt.year >= 2021) & (df_muestra['Date'].dt.year <= 2023)
        
        print("  [CHECK 2.3] Estandarizando basado EXCLUSIVAMENTE en datos de entrenamiento (2021-2023)...")
        for feature in features_req:
            if feature not in df_muestra.columns:
                continue # Falla en el check final si no existe
                
            media_train = df_muestra.loc[mask_train, feature].mean()
            std_train = df_muestra.loc[mask_train, feature].std()
            
            if pd.isna(std_train) or std_train == 0:
                std_train = 1e-8
                
            df_muestra[feature] = (df_muestra[feature] - media_train) / std_train

        df_muestra.to_csv(ruta_salida, index=False)
        print(f"  [CHECK 2.3] Guardado dataset listo para backtest en: {ruta_salida}")

        # ---------------------------------------------------------
        # PASO 2.4: Reporte Final de Compatibilidad
        # ---------------------------------------------------------
        print("\n" + "="*60)
        print("REPORTE DE COMPATIBILIDAD FINAL")
        print("="*60)
        
        errores = []
        
        # Check Tickers
        if df_muestra['Ticker'].nunique() != empresas_req:
            errores.append(f"- CARDINALIDAD: El modelo espera {empresas_req} acciones, hay {df_muestra['Ticker'].nunique()}.")
        else:
            print(f"✔ Cardinalidad correcta: {empresas_req} acciones.")

        # Check Features
        faltantes = [f for f in features_req if f not in df_muestra.columns]
        if faltantes:
            errores.append(f"- FEATURES: Faltan variables requeridas por la red: {faltantes}")
        else:
            print(f"✔ Features correctas: Las {len(features_req)} variables están presentes.")

        # Check Target
        target_col = config.get("TARGET_PRICE_COL", "Close")
        if target_col not in df_muestra.columns:
            errores.append(f"- COLUMNA OBJETIVO: Falta '{target_col}'.")
        else:
            print(f"✔ Columna de precio objetivo ('{target_col}') presente.")

        print("-" * 60)
        if not errores:
            print("✅ VEREDICTO: EL DATASET ES 100% COMPATIBLE.")
            print("El proceso ha finalizado con éxito. Ya puedes lanzar tu backtest de 2024-2025.")
        else:
            print("❌ VEREDICTO: EL DATASET NO SE PUEDE BACKTESTEAR. MOTIVOS:")
            for err in errores:
                print(err)
        print("="*60 + "\n")

    except Exception as e:
        logging.error(f"Error en Fase 2: {e}")

# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    # --- RUTAS DE ENTRADA (S&P 500) ---
    ARCHIVO_COTIZACIONES  = "cotiz_SP_technical_complete.csv"
    ARCHIVO_FUNDAMENTALES = "Analisis_Fundamental/S&P_v2/fundamentales_limpios.csv"
    
    # --- RUTAS INTERMEDIAS Y DE SALIDA ---
    DIR_SALIDA = "GA_Env/data"
    os.makedirs(DIR_SALIDA, exist_ok=True)
    
    DATASET_SP500_COMPLETO = f"{DIR_SALIDA}/dataset_S&P500_completo_v2_tecnicos.csv"
    DATASET_SP500_FINAL    = f"{DIR_SALIDA}/sp500_estandarizado_backtest_v2_tecnicos.csv"
    
    # --- RUTA DE TU CONFIGURACIÓN DEL MODELO ---
    RUTA_JSON_MODELO       = "GA_Env/experimentos/GA_MultiStock_v27/best_config_ga.json"
    
    # Ejecutamos Fase 1 (Generación)
    generar_dataset_completo(
        archivo_cotizaciones  = ARCHIVO_COTIZACIONES,
        archivo_fundamentales = ARCHIVO_FUNDAMENTALES,
        ruta_salida_completo  = DATASET_SP500_COMPLETO
    )
    
    # Ejecutamos Fase 2 (Ajuste para el modelo)
    preparar_y_validar_sp500(
        ruta_entrada = DATASET_SP500_COMPLETO,
        ruta_salida  = DATASET_SP500_FINAL,
        ruta_json    = RUTA_JSON_MODELO,
        semilla      = 123
    )