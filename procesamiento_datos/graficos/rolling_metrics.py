#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANÁLISIS DINÁMICO ANUAL (ROLLING METRICS)
Analiza la evolución temporal de la Rentabilidad, Volatilidad, Ratio de Sharpe 
y Maximum Drawdown año a año para cada activo, Índice y Sector.

Justificación:
1. Evolución del Sharpe por Índice: Permite evaluar la eficiencia relativa a lo largo del tiempo.
2. Heatmap de Sharpe por Sector: Facilita la detección de la rotación sectorial y consistencia.
3. Evolución de la Volatilidad por Sector: Fundamental para monitorear el riesgo estructural y 
   la sensibilidad a shocks externos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# ==============================================================================
# 0. INICIALIZACIÓN Y CARPETAS
# ==============================================================================
tiempo_inicio = time.time()
print("Iniciando Análisis Dinámico Anual (Rolling Metrics)...")

# Estructura de carpetas
dir_base = "Analisis_Dinamico_Anual"
os.makedirs(dir_base, exist_ok=True)
print(f"Directorio verificado/creado: ./{dir_base}/")

# Parámetros de cálculo
TASA_LIBRE_RIESGO = 0.02  # 2% anual

# ==============================================================================
# 1. CÁLCULO DE MÉTRICAS ROLLING (POR AÑO)
# ==============================================================================
print("1. Cargando datos y segmentando por años...")
df = pd.read_csv('cotizaciones_limpias.csv', parse_dates=['Date'])
df['Year'] = df['Date'].dt.year

# Asegurar nombres de columnas (mayúsculas/minúsculas)
col_indice = 'Indice' if 'Indice' in df.columns else 'indice'
col_sector = 'Sector' if 'Sector' in df.columns else 'sector'

anios = sorted(df['Year'].unique())
tickers = df['Ticker'].unique()

print("2. Calculando métricas anualizadas (Rentabilidad, Volatilidad, Sharpe, MDD)...")
resultados_anuales = []

for ticker in tickers:
    df_ticker = df[df['Ticker'] == ticker].sort_values('Date')
    if df_ticker.empty:
        continue
        
    indice_empresa = df_ticker[col_indice].iloc[0]
    sector_empresa = df_ticker[col_sector].iloc[0]
    
    for anio in anios:
        data_anio = df_ticker[df_ticker['Year'] == anio]
        
        # Necesitamos un mínimo de días cotizados para que el cálculo anual tenga sentido
        if len(data_anio) < 50:  
            continue
            
        precios = data_anio['Close']
        retornos = precios.pct_change().dropna()
        
        # Rentabilidad del año
        rentabilidad = (precios.iloc[-1] / precios.iloc[0]) - 1
        
        # Volatilidad del año
        volatilidad = retornos.std() * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe = (rentabilidad - TASA_LIBRE_RIESGO) / volatilidad if volatilidad > 0 else np.nan
        
        # Maximum Drawdown del año (NUEVA MÉTRICA AÑADIDA)
        max_acumulado = precios.cummax()
        drawdowns = (precios - max_acumulado) / max_acumulado
        mdd = drawdowns.min()
            
        resultados_anuales.append({
            'Ticker': ticker,
            col_indice: indice_empresa,
            col_sector: sector_empresa,
            'Año': anio,
            'Rentabilidad': rentabilidad,
            'Volatilidad': volatilidad,
            'Sharpe': sharpe,
            'Max_Drawdown': mdd
        })

# Convertimos a DataFrame
df_anual = pd.DataFrame(resultados_anuales)

# Guardamos el formato largo (útil para Tableau/PowerBI)
ruta_largo = os.path.join(dir_base, 'metricas_anuales_largo.csv')
df_anual.to_csv(ruta_largo, index=False)

# Pivotamos para formato ancho (Comparativa visual en Excel)
df_pivot = df_anual.pivot_table(
    index=['Ticker', col_indice, col_sector], 
    columns='Año', 
    values=['Sharpe', 'Volatilidad', 'Max_Drawdown']
)
df_pivot.columns = [f'{metrica}_{int(anio)}' for metrica, anio in df_pivot.columns]
ruta_ancho = os.path.join(dir_base, 'comparativa_anual_metricas.csv')
df_pivot.reset_index().to_csv(ruta_ancho, index=False)

print(f"Archivos de datos guardados en ./{dir_base}/")

# ==============================================================================
# 2. VISUALIZACIÓN DINÁMICA (ÍNDICES Y SECTORES)
# ==============================================================================
print("3. Generando visualizaciones del análisis dinámico...")

# Agrupaciones Macro
resumen_indice = df_anual.groupby(['Año', col_indice])[['Sharpe', 'Volatilidad', 'Rentabilidad', 'Max_Drawdown']].mean().reset_index()
resumen_sector = df_anual.groupby(['Año', col_sector])[['Sharpe', 'Volatilidad', 'Rentabilidad', 'Max_Drawdown']].mean().reset_index()

# Guardar los resúmenes estadísticos
resumen_indice.to_csv(os.path.join(dir_base, 'estudio_anual_por_indice.csv'), index=False)
resumen_sector.to_csv(os.path.join(dir_base, 'estudio_anual_por_sector.csv'), index=False)

# --- GRÁFICO 1: EVOLUCIÓN DEL SHARPE POR ÍNDICE ---
plt.figure(figsize=(12, 6))
sns.lineplot(data=resumen_indice, x='Año', y='Sharpe', hue=col_indice, marker='o', linewidth=2.5)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.title('Evolución del Ratio de Sharpe Medio por Índice', fontsize=15)
plt.ylabel('Ratio de Sharpe Medio')
plt.grid(True, alpha=0.3)
plt.xticks(resumen_indice['Año'].unique())
plt.savefig(os.path.join(dir_base, 'evolucion_sharpe_indice.png'), dpi=300)
plt.close()

# --- GRÁFICO 2: HEATMAP DE SHARPE POR SECTOR Y AÑO ---
pivot_sector_sharpe = resumen_sector.pivot(index=col_sector, columns='Año', values='Sharpe')
plt.figure(figsize=(14, 8))
sns.heatmap(pivot_sector_sharpe, annot=True, cmap='RdYlGn', center=0, fmt=".2f", linewidths=.5)
plt.title('Rotación Sectorial: Ratio de Sharpe por Sector y Año', fontsize=16)
plt.xlabel('Año')
plt.ylabel('Sector Económico')
plt.tight_layout()
plt.savefig(os.path.join(dir_base, 'heatmap_sharpe_sectorial.png'), dpi=300)
plt.close()

# --- GRÁFICO 3: EVOLUCIÓN DE LA VOLATILIDAD POR SECTOR ---
plt.figure(figsize=(14, 7))
sns.lineplot(data=resumen_sector, x='Año', y='Volatilidad', hue=col_sector, marker='s')
plt.title('Evolución de la Volatilidad Media por Sector (Riesgo Estructural)', fontsize=15)
plt.ylabel('Volatilidad Anualizada Media')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(dir_base, 'evolucion_volatilidad_sectorial.png'), dpi=300)
plt.close()

# --- GRÁFICO 4: HEATMAP DE MAXIMUM DRAWDOWN POR SECTOR Y AÑO (NUEVO) ---
pivot_sector_mdd = resumen_sector.pivot(index=col_sector, columns='Año', values='Max_Drawdown')
plt.figure(figsize=(14, 8))
# Usamos un mapa de calor inverso (Rojo para caídas más fuertes, Verde para caídas suaves)
sns.heatmap(pivot_sector_mdd, annot=True, cmap='RdYlGn', fmt=".1%", linewidths=.5)
plt.title('Mapa de Riesgo Extremo: Maximum Drawdown Medio por Sector y Año', fontsize=16)
plt.xlabel('Año')
plt.ylabel('Sector Económico')
plt.tight_layout()
plt.savefig(os.path.join(dir_base, 'heatmap_mdd_sectorial.png'), dpi=300)
plt.close()

# ==============================================================================
# 3. FIN DEL CRONÓMETRO
# ==============================================================================
tiempo_fin = time.time()
tiempo_total = tiempo_fin - tiempo_inicio

horas = int(tiempo_total // 3600)
minutos = int((tiempo_total % 3600) // 60)
segundos = int(tiempo_total % 60)

print("-" * 50)
print(f"Análisis Dinámico Anual completado ")
print(f"⏱️ Tiempo total de ejecución: {horas} horas, {minutos} minutos y {segundos} segundos.")
print("-" * 50)