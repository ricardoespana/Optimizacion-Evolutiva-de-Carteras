#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EXTRACCIÓN AUTOMATIZADA DE FUNDAMENTALES (PARAMETRIZADA Y ROBUSTA) - RANGO 2020-2025
Clasifica el universo usando TRBC (LSEG) y extrae métricas adaptadas 
a la realidad contable de cada sector. 
Fijado: Extracción estricta desde 2020 a 2025 para el modelo robusto.
"""

import lseg.data as ld
import pandas as pd
import numpy as np
import os
import time

# ==============================================================================
# 0. CONFIGURACIÓN GLOBAL (PARÁMETROS DE ENTRADA Y SALIDA)
# ==============================================================================
# Ruta base absoluta para evitar problemas con Spyder/Jupyter
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # 1. Parámetros Temporales y de Extracción
    START_YEAR = 2020  
    END_YEAR = 2025
    FRECUENCIA = "FQ"  # 'FQ' = Trimestral, 'FY' = Anual

    # 2. Rutas y Archivos
    DIR_SALIDA = os.path.join(BASE_DIR, "Analisis_Fundamental")
    os.makedirs(DIR_SALIDA, exist_ok=True)
    
    ARCHIVO_UNIVERSO = os.path.join(BASE_DIR, "df_u_lseg.csv")
    ARCHIVO_PANEL_FINAL = "fundamentales_universo.csv" 
    ARCHIVO_AUDITORIA = "Auditoria_Clasificacion_TRBC.csv"


# ==============================================================================
# 1. DICCIONARIOS DE EXTRACCIÓN ESTANDARIZADOS 
# ==============================================================================
CAMPOS_COMUNES = {
    "P_E": "TR.PE",                                        
    "Rentabilidad_Dividendo": "TR.DividendYield",
    "Capitalizacion_Bursatil": "TR.CompanyMarketCap",
    "Beneficio_Neto": "TR.NetIncomeInclExtraBeforeDistributions",
    "Activos_Totales": "TR.TotalAssets",
    "ROE":"TR.ROE",
    "Price-to-book value": "TR.PriceToBVPerShare",
    "Total_Debt_To_Equity": "TR.TotalDebtToEv" # Corregido typo: Tota_ -> Total_
}

CAMPOS_CORPORATE = {
    **CAMPOS_COMUNES,
    "Ingresos": "TR.Revenue",
    "EBITDA": "TR.EBITDA",
    "EBIT": "TR.EBIT",
    "Deuda_Total": "TR.TotalDebtOutstanding",
    "Operating_Margin":"TRF.F.OpMargPct",
    "Total_Debt_To_EBITDA": "TR.H.TotalDebtToEBITDA"
}

CAMPOS_BANCA = {
    **CAMPOS_COMUNES,
    "Ingresos_Intereses_Netos": "TR.F.IntrExpnNetOfIntrInc",
    "Gastos_Operativos": "TR.F.OpExpnTot",
    "Resultado_Operativo": "TR.F.intrDivIncFinTot",
    "Deuda_Total": "TR.F.LoansRcvblTot", 
    "Efficiency_Ratio":"TR.F.EfficiencyRatioPct",
    "Ratio_Tier_1":"TR.Tier1CapitalRatioActual"
}

CAMPOS_SEGUROS = {
    **CAMPOS_COMUNES,
    "Ingresos_Primas": "TR.F.NetPremEarned",          
    "Deuda_Total": "TR.F.DebtTot",
    "Gastos_Siniestros": "TR.F.InsurClaimsRsrvTot",   
    "Reservas_Tecnicas": "TR.F.InsurRsrvTot",        
    "Inversiones_Totales": "TR.F.InvstInclLoansTot",  
    "Solvency_Ratio":    "TR.F.SolvRatioPct",
    "Combined_Ratio":"TR.F.CombRatioInsurPct"
}

# ── MOLDE MAESTRO PARA EVITAR DESPLAZAMIENTO DE COLUMNAS (ACTUALIZADO) ──
COLUMNAS_MAESTRAS = [
    "Date", "Instrument", "Ticker", "Nombre", "Indice", "TRBC_Sector", "Grupo_Contable",
    # Comunes
    "P_E", "Rentabilidad_Dividendo", "Capitalizacion_Bursatil", "Beneficio_Neto", "Activos_Totales", 
    "ROE", "Price-to-book value", "Total_Debt_To_Equity",
    # Corporate
    "Ingresos", "EBITDA", "EBIT", "Deuda_Total", "Deuda_Neta", "Operating_Margin", "Total_Debt_To_EBITDA",
    # Banca
    "Ingresos_Intereses_Netos", "Gastos_Operativos", "Resultado_Operativo", "Depositos_Totales", 
    "Prestamos_Netos", "Efficiency_Ratio", "Ratio_Tier_1",
    # Seguros
    "Ingresos_Primas", "Gastos_Siniestros", "Reservas_Tecnicas", "Inversiones_Totales", 
    "Solvency_Ratio", "Combined_Ratio"
]


# ==============================================================================
# 2. MOTOR DE EXTRACCIÓN Y CONSTRUCCIÓN
# ==============================================================================
def descargar_por_lotes(rics, codigos_lseg, tamano_lote=20):
    """Descarga datos de LSEG dividiendo los RICs en lotes para evitar Timeouts."""
    datos_completos = []
    
    for i in range(0, len(rics), tamano_lote):
        lote_rics = rics[i:i + tamano_lote]
        intentos = 0
        exito = False
        
        while intentos < 3 and not exito:
            try:
                print(f"      - Lote {i//tamano_lote + 1} ({len(lote_rics)} RICs)...", end=" ", flush=True)
                df_lote = ld.get_data(
                    universe=lote_rics,
                    fields=codigos_lseg,
                    parameters={
                        "SDate": f"{Config.START_YEAR}-01-01",
                        "EDate": f"{Config.END_YEAR}-12-31",
                        "Frq": Config.FRECUENCIA
                    }
                )
                
                if df_lote is not None and not df_lote.empty:
                    # [!] CORRECCIÓN CLAVE: Eliminar duplicados INMEDIATAMENTE tras la descarga
                    df_lote = df_lote.loc[:, ~df_lote.columns.duplicated()]
                    datos_completos.append(df_lote)
                    
                print("OK")
                exito = True
                time.sleep(1)  # Pausa de cortesía para la API
                
            except Exception as e:
                intentos += 1
                print(f"Fallo (Intento {intentos}/3): {e}")
                time.sleep(4)
                
    if datos_completos:
        return pd.concat(datos_completos, axis=0, ignore_index=True)
    return pd.DataFrame()


def construir_panel_fundamental(df_universo):
    lista_dfs = []
    
    # Generar la columna Grupo_Contable de forma automática
    if 'Grupo_Contable' not in df_universo.columns and 'Industria' in df_universo.columns:
        df_universo['Grupo_Contable'] = 'CORPORATE'
        mascara_bancos = df_universo['Industria'].str.contains('Banks', case=False, na=False)
        df_universo.loc[mascara_bancos, 'Grupo_Contable'] = 'BANCA'
        mascara_seguros = df_universo['Industria'].str.contains('Insurance', case=False, na=False)
        df_universo.loc[mascara_seguros, 'Grupo_Contable'] = 'SEGUROS'

    grupos = df_universo.groupby('Grupo_Contable')
    
    for grupo, df_grupo in grupos:
        rics = df_grupo['RIC'].tolist() if 'RIC' in df_grupo.columns else df_grupo['Ticker'].tolist()
        
        if grupo in ['BANCA', 'Banks']:
            campos_dict = CAMPOS_BANCA
        elif grupo in ['SEGUROS', 'Insurance']:
            campos_dict = CAMPOS_SEGUROS
        else:
            campos_dict = CAMPOS_CORPORATE
            
        codigos_lseg = list(campos_dict.values())
        mapeo_renombres = {v: k for k, v in campos_dict.items()}
        
        print(f"\n   -> Extrayendo {len(rics)} RICs para el grupo contable: {grupo} (Rango {Config.START_YEAR}-{Config.END_YEAR})")
        
        try:
            # Reemplazamos la llamada directa por la función de lotes
            df_data = descargar_por_lotes(rics, codigos_lseg, tamano_lote=25)
            
            if df_data is not None and not df_data.empty:
                # 1. Renombrar las columnas devueltas por LSEG
                nuevas_cols = {}
                
                for col in df_data.columns:
                    c_low = col.lower()
                    if 'instrument' in c_low or 'ric' in c_low:
                        nuevas_cols[col] = 'Instrument'
                    elif 'date' in c_low or 'period' in c_low:
                        nuevas_cols[col] = 'Date'
                        
                cols_datos = [c for c in df_data.columns if c not in nuevas_cols.keys()]
                
                if len(cols_datos) == len(codigos_lseg):
                    for i, col in enumerate(cols_datos):
                        nuevas_cols[col] = mapeo_renombres[codigos_lseg[i]]
                else:
                    for col in cols_datos:
                        c_low = col.lower()
                        # Base
                        if 'earnings' in c_low or 'p/e' in c_low: nuevas_cols[col] = 'P_E'
                        elif 'dividend' in c_low: nuevas_cols[col] = 'Rentabilidad_Dividendo'
                        elif 'market cap' in c_low: nuevas_cols[col] = 'Capitalizacion_Bursatil'
                        elif 'net income' in c_low: nuevas_cols[col] = 'Beneficio_Neto'
                        elif 'assets' in c_low: nuevas_cols[col] = 'Activos_Totales'
                        # Comunes Nuevas
                        elif 'roe' in c_low or 'return on equity' in c_low: nuevas_cols[col] = 'ROE'
                        elif 'book value' in c_low or 'bv' in c_low.split(): nuevas_cols[col] = 'Price-to-book value'
                        elif 'debt to ev' in c_low or 'debt to eq' in c_low: nuevas_cols[col] = 'Total_Debt_To_Equity'
                        # Corporate
                        elif 'revenue' in c_low: nuevas_cols[col] = 'Ingresos'
                        elif 'ebitda' in c_low and 'debt' not in c_low: nuevas_cols[col] = 'EBITDA'
                        elif 'ebit' in c_low and 'debt' not in c_low: nuevas_cols[col] = 'EBIT'
                        elif 'operating margin' in c_low or 'opmarg' in c_low: nuevas_cols[col] = 'Operating_Margin'
                        elif 'debt to ebitda' in c_low: nuevas_cols[col] = 'Total_Debt_To_EBITDA'
                        elif 'debt' in c_low or 'loans' in c_low: nuevas_cols[col] = 'Deuda_Total'
                        # Banca
                        elif 'interest' in c_low or 'expn' in c_low: nuevas_cols[col] = 'Ingresos_Intereses_Netos'
                        elif 'operating expense' in c_low: nuevas_cols[col] = 'Gastos_Operativos'
                        elif 'operating income' in c_low: nuevas_cols[col] = 'Resultado_Operativo'
                        elif 'efficiency' in c_low: nuevas_cols[col] = 'Efficiency_Ratio'
                        elif 'tier 1' in c_low or 'tier1' in c_low: nuevas_cols[col] = 'Ratio_Tier_1'
                        elif 'deposits' in c_low: nuevas_cols[col] = 'Depositos_Totales'
                        # Seguros
                        elif 'premium' in c_low: nuevas_cols[col] = 'Ingresos_Primas'
                        elif 'claims' in c_low: nuevas_cols[col] = 'Gastos_Siniestros'
                        elif 'reserve' in c_low: nuevas_cols[col] = 'Reservas_Tecnicas'
                        elif 'investment' in c_low: nuevas_cols[col] = 'Inversiones_Totales'
                        elif 'solv' in c_low: nuevas_cols[col] = 'Solvency_Ratio'
                        elif 'comb' in c_low and 'ratio' in c_low: nuevas_cols[col] = 'Combined_Ratio'

                df_data = df_data.rename(columns=nuevas_cols)
                
                # 2. Añadir metadatos cruzando con el DF original
                cols_meta = [c for c in ['Ticker', 'Nombre', 'Indice', 'TRBC_Sector', 'Grupo_Contable'] if c in df_grupo.columns]
                df_data = df_data.merge(
                    df_grupo[cols_meta + ['RIC']], 
                    how='left', 
                    left_on='Instrument', 
                    right_on='RIC'
                )

                # 3. LIMPIEZA DEL BUG PER
                for c in df_data.columns:
                    if ('PER' in c or 'P/E' in c) and c != "P_E":
                        df_data = df_data.drop(columns=[c])

                # 4. SOLUCIÓN AL SHIFTING: Forzar el reindexado al molde maestro
                # (La deduplicación ya se hizo arriba, así que esto no fallará)
                df_estructurado = df_data.reindex(columns=COLUMNAS_MAESTRAS)
                lista_dfs.append(df_estructurado)
                
        except Exception as e:
            print(f"   [!] Error consolidando grupo {grupo}: {e}")

    if not lista_dfs:
        return pd.DataFrame()
        
    df_panel_final = pd.concat(lista_dfs, axis=0, ignore_index=True)
    
    cols_fijas = ["Date", "Instrument", "Ticker", "Nombre", "Indice", "TRBC_Sector", "Grupo_Contable"]
    todas_columnas = cols_fijas + [c for c in df_panel_final.columns if c not in cols_fijas]
    df_panel_final = df_panel_final[[c for c in todas_columnas if c in df_panel_final.columns]]
    
    if 'Ticker' in df_panel_final.columns and 'Date' in df_panel_final.columns:
        df_panel_final = df_panel_final.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
        
    return df_panel_final


# ==============================================================================
# 5. EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":
    from lseg_config import open_session, close_session

    tiempo_inicio = time.time()
    print("⏳ Iniciando Extracción y Clasificación Automática de Fundamentales (Rango 2020-2025)...")

    if not os.path.exists(Config.ARCHIVO_UNIVERSO):
        print(f"❌ ERROR: No se encuentra el archivo de universo base en: {Config.ARCHIVO_UNIVERSO}")
        import sys; sys.exit(1)

    print(f"1. Cargando descriptivos locales ({Config.ARCHIVO_UNIVERSO})...")
    df_universo = pd.read_csv(Config.ARCHIVO_UNIVERSO)

    try:
        print("🔌 Abriendo sesión en LSEG Workspace...")
        open_session()
        df_panel = construir_panel_fundamental(df_universo)
    except Exception as e:
        print(f"❌ Error crítico en el pipeline: {str(e)}")
        df_panel = pd.DataFrame()
    finally:
        print("🔌 Cerrando sesión en LSEG Workspace...")
        close_session()

    if not df_panel.empty:
        ruta_final = os.path.join(Config.DIR_SALIDA, Config.ARCHIVO_PANEL_FINAL)
        df_panel.to_csv(ruta_final, index=False)
        print(f"\n✅ Panel consolidado con éxito en: {ruta_final}")
        
    t_fin = time.time()
    mins = int((t_fin - tiempo_inicio) // 60)
    segs = int((t_fin - tiempo_inicio) % 60)
    print(f"🏁 Proceso completado en {mins}m {segs}s.")