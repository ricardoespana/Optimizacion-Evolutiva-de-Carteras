#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# PASO 1: CONFIGURACIÓN Y CARGA DE DATOS
# ==============================================================================
print("1. Configurando entorno y cargando datos...")

# Creamos la carpeta de salida si no existe
carpeta_salida = "reporte_eda/news_eleconomista" #FIJARSE EN EL DATASET A QUE PERIODICO PERTENECE
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

# Nombre del archivo generado en el paso de scraping
archivo_noticias = "dataset_2021_2026_eleconomista.csv" 
df = pd.read_csv(archivo_noticias)

# ==============================================================================
# PASO 2: LIMPIEZA BÁSICA Y TRANSFORMACIÓN DE TIPOS
# ==============================================================================
print("2. Limpiando el dataset...")

# 2.1 Eliminar filas que no tengan texto
df = df.dropna(subset=['Texto_Analizar'])

# 2.2 Convertir la columna de fecha a un objeto datetime real
df['Fecha_Publicacion'] = pd.to_datetime(df['Fecha_Publicacion'], errors='coerce', utc=True)

# ── FILTRADO TEMPORAL 2021-2026 ──
# Mantenemos solo filas donde el año sea igual o superior a 2021
df = df[df['Fecha_Publicacion'].dt.year >= 2021].copy()
print(f"   -> Filtrado aplicado: Dataset reducido a {len(df)} noticias desde 2021.")
# ────────────────────────────────

# 2.3 Crear columnas auxiliares de tiempo para agrupar luego
df['Año'] = df['Fecha_Publicacion'].dt.year
df['Mes_Año'] = df['Fecha_Publicacion'].dt.to_period('M')

# 2.4 Detección y eliminación de duplicados exactos 
duplicados_iniciales = df.duplicated(subset=['Titular', 'Ticker']).sum()
df = df.drop_duplicates(subset=['Titular', 'Ticker'], keep='first')
print(f"   -> Eliminados {duplicados_iniciales} registros duplicados.")
# ==============================================================================
# PASO 3: INGENIERÍA DE CARACTERÍSTICAS TEXTUALES (Feature Engineering)
# ==============================================================================
print("3. Calculando métricas estructurales del texto...")

# 3.1 Contar el número de palabras separando por espacios
df['Num_Palabras'] = df['Texto_Analizar'].apply(lambda x: len(str(x).split()))

# 3.2 Contar el número de caracteres totales
df['Num_Caracteres'] = df['Texto_Analizar'].apply(lambda x: len(str(x)))

# ==============================================================================
# PASO 4: GENERACIÓN DE REPORTES VISUALES (Las Gráficas)
# ==============================================================================
print("4. Generando visualizaciones clave...")

# Establecemos el estilo de seaborn para que sea elegante
sns.set_theme(style="whitegrid")

# --- GRÁFICO A: Distribución del Volumen por Índice y Sector ---
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.countplot(data=df, x='Indice', order=df['Indice'].value_counts().index, palette='Blues_r', ax=axes[0])
axes[0].set_title("Volumen de Noticias por Índice", fontsize=14)
axes[0].set_ylabel("Nº de Noticias")

sns.countplot(data=df, y='Sector', order=df['Sector'].value_counts().index, palette='crest', ax=axes[1])
axes[1].set_title("Volumen de Noticias por Sector Económico", fontsize=14)
axes[1].set_xlabel("Nº de Noticias")

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "01_volumen_indice_sector.png"), dpi=300)
plt.close()

# --- GRÁFICO B: Estabilidad Temporal del Scraping ---
plt.figure(figsize=(15, 5))
noticias_por_mes = df.groupby('Mes_Año').size()
# Convertimos el índice a string para evitar errores en matplotlib
noticias_por_mes.index = noticias_por_mes.index.astype(str) 

plt.plot(noticias_por_mes.index, noticias_por_mes.values, color='darkred', marker='o', linewidth=2)
plt.title("Evolución Temporal de Publicaciones (Verificación de Estabilidad)", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.ylabel("Noticias descargadas")
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "02_estabilidad_temporal.png"), dpi=300)
plt.close()

# --- GRÁFICO C: Distribución de Longitud para  ---
plt.figure(figsize=(10, 6))
sns.histplot(df['Num_Palabras'], bins=50, kde=True, color='teal')


plt.title("Distribución de la Longitud del Texto de las Noticias", fontsize=14)
plt.xlabel("Número de Palabras")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "03_longitud_PALABRAS.png"), dpi=300)
plt.close()

# --- GRÁFICO D: Desequilibrio de Datos (Top vs Bottom Empresas) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

top_15 = df['Ticker'].value_counts().head(15)
sns.barplot(x=top_15.index, y=top_15.values, palette='Greens_r', ax=axes[0])
axes[0].set_title("Top 15 Empresas con MÁS Noticias (Saturación)", fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

bottom_15 = df['Ticker'].value_counts().tail(15)
sns.barplot(x=bottom_15.index, y=bottom_15.values, palette='Reds_r', ax=axes[1])
axes[1].set_title("Top 15 Empresas con MENOS Noticias (Sparsity)", fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "04_desequilibrio_empresas.png"), dpi=300)
plt.close()

# ==============================================================================
# PASO 5: REPORTE ESTADÍSTICO EN CONSOLA
# ==============================================================================
print("5. Análisis completado. Resumen Estadístico:")
print("-" * 50)
print(f"Total de noticias válidas procesadas: {len(df)}")
print(f"Empresas únicas con al menos 1 noticia: {df['Ticker'].nunique()}")
print(f"Longitud Media del Texto: {df['Num_Palabras'].mean():.0f} palabras")
print(f"Noticias por encima del límite FinBERT (>400 palabras): {(df['Num_Palabras'] > 400).sum()} ({(df['Num_Palabras'] > 400).sum() / len(df) * 100:.1f}%)")
print(f"Noticias extremadamente cortas (<50 palabras): {(df['Num_Palabras'] < 50).sum()}")
print("-" * 50)
print(f"Todas las gráficas se han guardado en: {carpeta_salida}")