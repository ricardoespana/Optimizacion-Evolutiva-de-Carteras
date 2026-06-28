#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JUSTIFICACIÓN DEL ANÁLISIS EXPLORATORIO DE DATOS (EDA):
El objetivo de este bloque es caracterizar la naturaleza estadística de las series temporales de precios y 
volúmenes. 
1. Histogramas de Retornos: Permiten verificar visualmente la presencia de 'colas pesadas' y la desviación 
   respecto a la distribución normal (curtosis), algo crítico para el cálculo del VaR.
2. Boxplots de Volatilidad: Facilitan la identificación de 'outliers' o días de pánico de mercado que 
   afectan de forma desigual a cada activo.
3. Análisis de Autocorrelación: Crucial para entender si los retornos pasados influyen en los futuros 
   (efecto memoria) y validar la eficiencia del mercado.
4. Evolución del Volumen vs Precio: Identifica la liquidez de los activos, un factor de riesgo actuarial 
   que determina la facilidad de liquidar posiciones en momentos de crisis.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EDA AVANZADO: MACRO (Universo) y MICRO (Activos Individuales)
-------------------------------------------------------------
Genera un análisis exploratorio de datos completo y lo organiza 
en un árbol de directorios jerárquico.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

# ==============================================================================
# 1. CREACIÓN DE LA ESTRUCTURA DE DIRECTORIOS
# ==============================================================================
print("1. Configurando el árbol de directorios...")

# Definición de rutas
DIR_BASE = "reporte_eda"
DIR_MACRO = os.path.join(DIR_BASE, "macro_universo")
DIR_MICRO = os.path.join(DIR_BASE, "micro_aciones")

# Creación de carpetas principales
os.makedirs(DIR_MACRO, exist_ok=True)
os.makedirs(DIR_MICRO, exist_ok=True)

# ==============================================================================
# 2. CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================
print("2. Cargando datos para el EDA...")
# Asegúrate de que el archivo contiene las columnas: Date, Ticker, Close, Volume, (y opcional 'indice')
df = pd.read_csv('cotizaciones_limpias.csv', parse_dates=['Date'])
df = df.sort_values(by=['Ticker', 'Date'])

# Calculamos retornos logarítmicos
df['Log_Ret'] = df.groupby('Ticker')['Close'].transform(lambda x: np.log(x / x.shift(1)))

# Calculamos volatilidad móvil de 20 días
df['Vol_20d'] = df.groupby('Ticker')['Log_Ret'].transform(lambda x: x.rolling(20).std() * np.sqrt(252))

# ==============================================================================
# 3. ANÁLISIS MACRO (UNIVERSO COMPLETO)
# ==============================================================================
print("3. Ejecutando Análisis MACRO...")

# 3.1 Estadísticas Descriptivas Generales
eda_summary = df.groupby('Ticker')[['Close', 'Volume', 'Log_Ret']].agg(['mean', 'std', 'min', 'max', 'skew'])
eda_summary.to_csv(os.path.join(DIR_MACRO, '01_estadisticos_universo.csv'))

# 3.2 Mapa de Disponibilidad (Datos Faltantes)
print("   -> Generando Heatmap de disponibilidad...")
# 1. Preparar la tabla de disponibilidad (NaN = ausente)
# pivot_availability muestra True donde hay datos y False donde hay NaN
pivot_availability = df.pivot(index='Date', columns='Ticker', values='Close')

# 2. Conversión forzosa del índice a DatetimeIndex
pivot_availability.index = pd.to_datetime(pivot_availability.index, errors='coerce')
pivot_availability = pivot_availability[pivot_availability.index.notna()]

# 3. Cálculo de los puntos de cambio de año para las etiquetas (Eje Y)
# Obtenemos el año de cada fila del índice
anios = pivot_availability.index.year
# Detectamos en qué índice (fila) cambia el año respecto al anterior
cambios_anio = np.where(anios != anios.to_series().shift(1))[0]
# Extraemos el valor del año en esas posiciones
etiquetas_anio = anios[cambios_anio].astype(int)

# 4. Generación del gráfico
plt.figure(figsize=(16, 8))
ax = sns.heatmap(pivot_availability.isnull(), cbar=False, yticklabels=False, cmap='viridis')

# 5. Aplicar las etiquetas anuales al eje Y
ax.set_yticks(cambios_anio)
ax.set_yticklabels(etiquetas_anio, rotation=0, fontsize=10)

plt.title('Mapa de Disponibilidad de Datos (Amarillo = Datos faltantes)', fontsize=14, fontweight='bold')
plt.xlabel('Empresas (Tickers)', fontsize=12)
plt.ylabel('Año', fontsize=12)
plt.tight_layout()

# 6. Guardado
plt.savefig(os.path.join(DIR_MACRO, '02_heatmap_disponibilidad.png'), dpi=300)
plt.close()
print("✅ Mapa de disponibilidad generado con éxito con etiquetas anuales.")

# 3.3 Boxplot de Retornos y Dispersión
print("   -> Generando Boxplots de riesgo...")
plt.figure(figsize=(20, 8))
sns.boxplot(data=df, x='Ticker', y='Log_Ret', palette='Spectral')
plt.xticks(rotation=90, fontsize=8)
plt.title('Dispersión de Retornos y Outliers por Empresa (Universo Completo)', fontsize=14)
plt.ylabel('Retorno Logarítmico Diario')
plt.tight_layout()
plt.savefig(os.path.join(DIR_MACRO, '03_boxplot_retornos.png'), dpi=300)
plt.close()

# 3.4 Relación Volumen vs Volatilidad
print("   -> Generando Volumen vs Volatilidad...")
plt.figure(figsize=(12, 8))
# Usamos 'indice' para el color si existe, si no, un color plano
hue_col = 'indice' if 'indice' in df.columns else None
sns.scatterplot(data=df, x='Volume', y='Vol_20d', alpha=0.3, hue=hue_col)
plt.xscale('log') 
plt.title('Relación Volumen vs Volatilidad Anualizada (Proxy de Liquidez)', fontsize=14)
plt.xlabel('Volumen (Escala Log)')
plt.ylabel('Volatilidad Móvil 20d Anualizada')
plt.tight_layout()
plt.savefig(os.path.join(DIR_MACRO, '04_volumen_vs_volatilidad.png'), dpi=300)
plt.close()

# ==============================================================================
# 4. ANÁLISIS MICRO (POR CADA ACCIÓN)
# ==============================================================================
tickers = df['Ticker'].unique()
print(f"4. Ejecutando Análisis MICRO para {len(tickers)} activos. Esto puede tardar un momento...")

for idx, ticker in enumerate(tickers, 1):
    # Imprimir progreso cada 10 empresas
    if idx % 10 == 0 or idx == len(tickers):
        print(f"   -> Procesando {idx}/{len(tickers)}: {ticker}...")
        
    # Crear sub-subcarpeta para el Ticker
    DIR_ACTIVO = os.path.join(DIR_MICRO, ticker)
    os.makedirs(DIR_ACTIVO, exist_ok=True)
    
    # Extraer datos de la acción actual
    df_ticker = df[df['Ticker'] == ticker].dropna(subset=['Log_Ret'])
    
    if len(df_ticker) < 30:
        continue # Saltamos empresas con muy pocos datos
    
    # 4.1 Gráfico A: Evolución del Precio y Volumen
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    ax1.plot(df_ticker['Date'], df_ticker['Close'], color='black')
    ax1.set_title(f'{ticker}: Evolución del Precio y Volumen', fontsize=14)
    ax1.set_ylabel('Precio de Cierre')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(df_ticker['Date'], df_ticker['Volume'], color='gray', alpha=0.7)
    ax2.set_ylabel('Volumen')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_ACTIVO, f'{ticker}_01_precio_volumen.png'), dpi=150)
    plt.close()
    
    # 4.2 Gráfico B: Histograma de Retornos (Normalidad y Colas)
    plt.figure(figsize=(8, 6))
    sns.histplot(df_ticker['Log_Ret'], kde=True, color='skyblue', bins=50)
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1)
    plt.title(f'{ticker}: Distribución de Retornos Logarítmicos')
    plt.xlabel('Retorno Diario')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_ACTIVO, f'{ticker}_02_histograma_retornos.png'), dpi=150)
    plt.close()
    
    # 4.3 Gráfico C: Autocorrelación (ACF) 
    # Muestra si los retornos de ayer predicen los de hoy
    plt.figure(figsize=(10, 5))
    plot_acf(df_ticker['Log_Ret'], lags=20, alpha=0.05, ax=plt.gca())
    plt.title(f'{ticker}: Autocorrelación de Retornos (Lags 1-20)')
    plt.xlabel('Lags (Días)')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_ACTIVO, f'{ticker}_03_autocorrelacion.png'), dpi=150)
    plt.close()

print("\n" + "="*50)
print("¡PROCESO COMPLETADO EXITOSAMENTE!")
print("Revisa la carpeta 'reporte_eda' para explorar tus resultados.")
print("="*50)
