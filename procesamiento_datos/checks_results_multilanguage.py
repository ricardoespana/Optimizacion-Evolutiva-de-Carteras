#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANALISIS DE RESULTADOS NLP - NLPTOWN MULTILINGUAL (1-5 ESTRELLAS)
Analiza los resultados del modelo, genera graficos de distribucion.
Calcula la rentabilidad anualizada desde 'cotizaciones_limpias.csv'
y crea el scatter plot de Sentimiento vs Rentabilidad.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import os
import sys

# ==============================================================================
# 1. CONFIGURACION GLOBAL Y RUTAS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    MODELO_NOMBRE_CORTO = "nlptown_bert"
    DIR_ENTRADA_SALIDA = os.path.join(BASE_DIR, "resultados_NLP", MODELO_NOMBRE_CORTO)
    ARCHIVO_ENTRADA = os.path.join(DIR_ENTRADA_SALIDA, f"{MODELO_NOMBRE_CORTO}_results.csv")
    ARCHIVO_REPORTE = os.path.join(DIR_ENTRADA_SALIDA, "Reporte_Resultados_NLPTown.txt")
    
    # Archivo de cotizaciones para calcular rentabilidad
    ARCHIVO_COTIZACIONES = os.path.join(BASE_DIR, "cotizaciones_limpias.csv")

sns.set_theme(style="whitegrid", context="paper")

# Asegurar que el directorio de salida existe
if not os.path.exists(Config.DIR_ENTRADA_SALIDA):
    os.makedirs(Config.DIR_ENTRADA_SALIDA)

# ==============================================================================
# 2. CARGA DE DATOS Y CÁLCULO DE RENTABILIDAD
# ==============================================================================
print("="*60)
print(f"INICIANDO ANALISIS DE RESULTADOS: {Config.MODELO_NOMBRE_CORTO}")
print("="*60)

if not os.path.exists(Config.ARCHIVO_ENTRADA):
    print(f"[ERROR FATAL] No se encontro '{Config.ARCHIVO_ENTRADA}'.")
    sys.exit(1)

# Cargar NLP
df = pd.read_csv(Config.ARCHIVO_ENTRADA)
df.columns = df.columns.str.strip()

# Cargar y calcular Cotizaciones
df_riesgo = pd.DataFrame()
if os.path.exists(Config.ARCHIVO_COTIZACIONES):
    print(f"-> Calculando rentabilidad desde: {Config.ARCHIVO_COTIZACIONES}")
    df_coti = pd.read_csv(Config.ARCHIVO_COTIZACIONES)
    df_coti.columns = df_coti.columns.str.strip()
    
    # Asegurar formato fecha y numérico
    df_coti['Date'] = pd.to_datetime(df_coti['Date'], errors='coerce')
    df_coti['Close'] = pd.to_numeric(df_coti['Close'], errors='coerce')
    df_coti = df_coti.dropna(subset=['Date', 'Close'])
    
    resultados_rent = []
    # Agrupar por Ticker
    for ticker, group in df_coti.groupby('Ticker'):
        group = group.sort_values('Date')
        if len(group) >= 2:
            first_price = group['Close'].iloc[0]
            last_price = group['Close'].iloc[-1]
            first_date = group['Date'].iloc[0]
            last_date = group['Date'].iloc[-1]
            
            years = (last_date - first_date).days / 365.25
            
            if years > 0 and first_price > 0 and last_price > 0:
                cagr = (last_price / first_price) ** (1 / years) - 1
                indice = group['Indice'].iloc[-1] if 'Indice' in group.columns else 'Desconocido'
                
                resultados_rent.append({
                    'Ticker': str(ticker).strip().upper(),
                    'Rentabilidad_Anualizada': cagr,
                    'Indice': str(indice).strip()
                })
                
    df_riesgo = pd.DataFrame(resultados_rent)
    print(f"   [EXITO] Rentabilidad calculada para {len(df_riesgo)} empresas.")
else:
    print(f"   [AVISO] No se encontro '{Config.ARCHIVO_COTIZACIONES}'.")

# Identificar columna Ticker en el df de NLP
col_ticker = None
posibles_nombres = ['Ticker', 'ticker', 'TICKER', 'Empresa', 'empresa', 'Activo', 'activo']
for col in df.columns:
    if col.strip() in posibles_nombres:
        col_ticker = col
        break

if col_ticker and not df_riesgo.empty:
    df[col_ticker] = df[col_ticker].astype(str).str.strip().str.upper()
    df = pd.merge(df, df_riesgo[['Ticker', 'Rentabilidad_Anualizada']], left_on=col_ticker, right_on='Ticker', how='left')
    print("   [EXITO] Rentabilidad anexada a las noticias de NLP.\n")

# Limpiar errores de NLP
df_validos = df[df['Estrellas'] != -1].copy()
errores = len(df) - len(df_validos)
total_noticias = len(df)

print(f"-> Total procesado: {total_noticias} | Validas: {len(df_validos)} | Errores: {errores}\n")

# ==============================================================================
# 3. GRAFICOS BÁSICOS (Distribución y Confianza)
# ==============================================================================
print("1. Generando Graficos de Distribucion y Confianza...")
# Grafico 1
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df_validos, x='Estrellas', order=[1, 2, 3, 4, 5], 
                   palette='viridis', hue='Estrellas', legend=False)
plt.title('Distribucion de Sentimiento Financiero (1 a 5 Estrellas)', fontsize=14, pad=15)
plt.ylabel('Cantidad de Noticias')
plt.xlabel('Calificacion (Estrellas)')
plt.tight_layout()
plt.savefig(os.path.join(Config.DIR_ENTRADA_SALIDA, '01_Distribucion_Estrellas.png'), dpi=300)

# Grafico 2 (Histograma)
plt.figure(figsize=(8, 5))
media_seg = df_validos['Seguridad_IA'].mean()
mediana_seg = df_validos['Seguridad_IA'].median()
sns.histplot(data=df_validos, x='Seguridad_IA', bins=20, kde=True, color='royalblue')
plt.axvline(media_seg, color='red', linestyle='--', label=f"Media: {media_seg:.3f}")
plt.axvline(mediana_seg, color='green', linestyle=':', label=f"Mediana: {mediana_seg:.3f}")
plt.title('Distribucion de Confianza (NLPTown)', fontsize=14, pad=15)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(Config.DIR_ENTRADA_SALIDA, '02_Histograma_Confianza_Multi.png'), dpi=300)

# Grafico 3 (Violin)
plt.figure(figsize=(8, 5))
sns.violinplot(data=df_validos, x='Estrellas', y='Seguridad_IA', order=[1, 2, 3, 4, 5], 
               palette='viridis', hue='Estrellas', legend=False, inner="quartile")
plt.title('Distribucion de Confianza según Calificacion', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(Config.DIR_ENTRADA_SALIDA, '03_Violin_Confianza_Estrellas.png'), dpi=300)

# ==============================================================================
# 4. SCATTER PLOT: SCORE POLARIZADO VS RENTABILIDAD
# ==============================================================================
print("2. Generando Scatter Plot de Rentabilidad...")

if col_ticker and not df_riesgo.empty:
    # Asegurar que Estrellas es numérico
    df_validos['Estrellas'] = pd.to_numeric(df_validos['Estrellas'], errors='coerce')
    df_temp = df_validos.dropna(subset=['Estrellas']).copy()
    
    # Calcular Score Polarizado
    df_temp['Score_Polarizado'] = (df_temp['Estrellas'] - 3) / 2
    
    # Agrupar por Ticker
    df_agrupado = df_temp.groupby(col_ticker).agg(
        Score_Medio_Empresa=('Score_Polarizado', 'mean'),
        Volumen_Noticias=('Estrellas', 'count')
    ).reset_index()
    
    # --- INICIO LIMPIEZA EXTREMA DE TICKERS ---
    # Limpiamos espacios, pasamos a mayúsculas y eliminamos sufijos comunes del mercado español
    df_agrupado['Ticker_Key'] = df_agrupado[col_ticker].astype(str).str.strip().str.upper()
    df_agrupado['Ticker_Key'] = df_agrupado['Ticker_Key'].str.replace('.MC', '', regex=False)
    
    df_riesgo['Ticker_Key'] = df_riesgo['Ticker'].astype(str).str.strip().str.upper()
    df_riesgo['Ticker_Key'] = df_riesgo['Ticker_Key'].str.replace('.MC', '', regex=False)
    
    # Diagnóstico visual por consola
    print(f"   [DEBUG] Ejemplos Tickers en NLP: {df_agrupado['Ticker_Key'].unique()[:6]}")
    print(f"   [DEBUG] Ejemplos Tickers en Cotizaciones: {df_riesgo['Ticker_Key'].unique()[:6]}")
    # --- FIN LIMPIEZA EXTREMA ---

    # Cruce definitivo
    df_scatter = pd.merge(df_agrupado, df_riesgo, on='Ticker_Key', how='inner')
    df_scatter = df_scatter.dropna(subset=['Indice', 'Rentabilidad_Anualizada', 'Score_Medio_Empresa'])
    
    if not df_scatter.empty:
        plt.figure(figsize=(10, 6))
        
        # Paleta segura
        colores_base = {'IBEX 35': 'green', 'IBEX_MEDIUM_CAP': 'blue', 'IBEX_SMALL_CAP': 'red'}
        paleta_dinamica = {idx: colores_base.get(idx, 'gray') for idx in df_scatter['Indice'].unique()}
        
        sns.scatterplot(
            data=df_scatter, x='Score_Medio_Empresa', y='Rentabilidad_Anualizada', 
            hue='Indice', palette=paleta_dinamica, s=150, alpha=0.8, edgecolor='black'
        )
        
        # Ejes y formato
        plt.axvline(0, color='gray', linestyle='--', alpha=0.6)
        plt.axhline(0, color='gray', linestyle='--', alpha=0.6)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        
        plt.title('Relacion Score Polarizado vs Rentabilidad Anualizada (2021-2025)', fontsize=14, pad=15)
        plt.xlabel('Score de Sentimiento (-1 Negativo a +1 Positivo)')
        plt.ylabel('Rentabilidad Anualizada (CAGR %)')
        plt.legend(title="Índice", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        ruta_scatter = os.path.join(Config.DIR_ENTRADA_SALIDA, '04_Scatter_Score_Rentabilidad.png')
        plt.savefig(ruta_scatter, dpi=300, bbox_inches='tight')
        print(f"   -> Scatter plot generado con exito. ({len(df_scatter)} empresas graficadas)")
        
        # Exportar los datos matematicos usados para auditar el grafico
        ruta_csv_datos = os.path.join(Config.DIR_ENTRADA_SALIDA, 'datos_del_scatterplot.csv')
        df_scatter.drop(columns=['Ticker_Key']).to_csv(ruta_csv_datos, index=False)
    else:
        print("   [AVISO] El grafico no se genero. Revisa las listas de [DEBUG] de arriba para ver por que no coinciden.")
else:
    print("   [AVISO] Falta archivo de cotizaciones o no hay columna Ticker.")
# ==============================================================================
# 5. REPORTE FINAL
# ==============================================================================
print("\n3. Guardando reporte de texto...")
texto_reporte = f"REPORTE NLP - NLPTOWN\n" + "="*40 + f"\nTotal procesado: {total_noticias}\nValidas: {len(df_validos)}\nErrores: {errores}\n\n"
texto_reporte += f"Confianza Media: {media_seg:.4f}\nConfianza Mediana: {mediana_seg:.4f}\n\n"
texto_reporte += "EJEMPLOS EXTREMOS:\n"

col_texto = 'Texto_Analizar' if 'Texto_Analizar' in df_validos.columns else 'Texto_BERT'
for est in [5, 1]:
    top_ejemplos = df_validos[df_validos['Estrellas'] == est].sort_values(by='Seguridad_IA', ascending=False).head(2)
    for _, fila in top_ejemplos.iterrows():
        txt = str(fila[col_texto]).replace('\n', ' ')[:100] + '...'
        texto_reporte += f"[Estrella {est} | Score {fila['Seguridad_IA']:.4f}] {txt}\n"

with open(Config.ARCHIVO_REPORTE, 'w', encoding='utf-8') as f:
    f.write(texto_reporte)

print("\n" + "="*60)
print(f"[EXITO] Analisis completado en la carpeta: {Config.DIR_ENTRADA_SALIDA}")
print("="*60)