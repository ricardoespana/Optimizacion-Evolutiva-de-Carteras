#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
ANALISIS DE RESULTADOS NLP - BARDSAI FINANCE
Genera graficos de distribucion, evalua la confianza de la red
y exporta un reporte en formato texto (.txt) con los resultados.

⏱️ Tiempo total de análisis NLP: 8 minutos y 12 segundos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ==============================================================================
# 1. CONFIGURACION GLOBAL Y RUTAS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    MODELO_NOMBRE_CORTO = "bardsai_finance"
    DIR_ENTRADA_SALIDA = os.path.join(BASE_DIR, "resultados_NLP", MODELO_NOMBRE_CORTO)
    ARCHIVO_ENTRADA = os.path.join(DIR_ENTRADA_SALIDA, f"{MODELO_NOMBRE_CORTO}_results.csv")
    ARCHIVO_REPORTE = os.path.join(DIR_ENTRADA_SALIDA, "Reporte_Resultados_BardsAI.txt")

sns.set_theme(style="whitegrid", context="paper")

# ==============================================================================
# 2. CARGA Y LIMPIEZA DE DATOS
# ==============================================================================
print("="*60)
print(f"INICIANDO ANALISIS DE RESULTADOS: {Config.MODELO_NOMBRE_CORTO}")
print("="*60)

if not os.path.exists(Config.ARCHIVO_ENTRADA):
    print(f"[ERROR] No se encontro '{Config.ARCHIVO_ENTRADA}'.")
    sys.exit(1)

df = pd.read_csv(Config.ARCHIVO_ENTRADA)

df_validos = df[df['Sentimiento'] != 'ERROR'].copy()
errores = len(df) - len(df_validos)
total_noticias = len(df)

print(f"-> Total procesado: {total_noticias} | Validas: {len(df_validos)} | Errores: {errores}\n")

# Iniciar la variable que almacenara el texto del reporte
texto_reporte = "REPORTE DE RESULTADOS NLP - BARDSAI FINANCE\n"
texto_reporte += "="*60 + "\n\n"
texto_reporte += f"1. RESUMEN DE EJECUCION\n"
texto_reporte += f"   - Total procesado: {total_noticias}\n"
texto_reporte += f"   - Noticias validas: {len(df_validos)}\n"
texto_reporte += f"   - Errores de lectura: {errores}\n\n"

# ==============================================================================
# 3. DISTRIBUCION DE SENTIMIENTOS
# ==============================================================================
print("1. Calculando Distribucion de Sentimientos...")
distribucion = df_validos['Sentimiento'].value_counts()
porcentajes = df_validos['Sentimiento'].value_counts(normalize=True) * 100

texto_reporte += "2. DISTRIBUCION DE SENTIMIENTOS\n"
for sentimiento, count in distribucion.items():
    linea = f"   - {sentimiento}: {count} noticias ({porcentajes[sentimiento]:.2f}%)"
    print(linea)
    texto_reporte += linea + "\n"

plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df_validos, x='Sentimiento', 
                   order=['Positivo', 'Negativo', 'Neutral'], 
                   palette='viridis')
plt.title('Distribucion de Sentimientos (Modelo BardsAI)', fontsize=14, pad=15)
plt.ylabel('Cantidad de Noticias')
plt.xlabel('Sentimiento')

for p in ax.patches:
    altura = p.get_height()
    if pd.notna(altura) and altura > 0:
        ax.text(p.get_x() + p.get_width()/2., altura + (altura*0.02), 
                f'{int(altura)}\n({(altura/len(df_validos))*100:.1f}%)', 
                ha="center", fontsize=10)

plt.tight_layout()
ruta_grafico_1 = os.path.join(Config.DIR_ENTRADA_SALIDA, '01_Distribucion_Sentimientos.png')
plt.savefig(ruta_grafico_1, dpi=300)

# ==============================================================================
# 4. ANALISIS DE CONFIANZA
# ==============================================================================
print("\n2. Analizando Confianza de la Red Neuronal...")
media_seguridad = df_validos['Seguridad_IA'].mean()
mediana_seguridad = df_validos['Seguridad_IA'].median()

print(f"   - Confianza Media:   {media_seguridad:.4f}")
print(f"   - Confianza Mediana: {mediana_seguridad:.4f}")

texto_reporte += "\n3. METRICAS DE CONFIANZA (Softmax)\n"
texto_reporte += f"   - Confianza Media: {media_seguridad:.4f}\n"
texto_reporte += f"   - Confianza Mediana: {mediana_seguridad:.4f}\n\n"

plt.figure(figsize=(8, 5))
sns.histplot(data=df_validos, x='Seguridad_IA', bins=20, kde=True, color='royalblue')
plt.axvline(media_seguridad, color='red', linestyle='--', label=f"Media: {media_seguridad:.3f}")
plt.axvline(mediana_seguridad, color='green', linestyle=':', label=f"Mediana: {mediana_seguridad:.3f}")
plt.title('Distribucion de Confianza (BardsAI)', fontsize=14, pad=15)
plt.xlabel('Probabilidad')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()

ruta_grafico_2 = os.path.join(Config.DIR_ENTRADA_SALIDA, '02_Histograma_Confianza.png')
plt.savefig(ruta_grafico_2, dpi=300)

# ==============================================================================
# 5. EXTRACCION DE EJEMPLOS Y GUARDADO DEL TXT
# ==============================================================================
print("\n3. Extrayendo Ejemplos Representativos...")
texto_reporte += "4. EJEMPLOS REPRESENTATIVOS (Alta Confianza)\n"

for categoria in ['Positivo', 'Negativo', 'Neutral']:
    encabezado = f"\n>> Top 2 ejemplos [{categoria.upper()}]:"
    print(encabezado)
    texto_reporte += encabezado + "\n"
    
    top_ejemplos = df_validos[df_validos['Sentimiento'] == categoria].sort_values(by='Seguridad_IA', ascending=False).head(2)
    for _, fila in top_ejemplos.iterrows():
        texto_completo = str(fila['Texto_Analizar']).replace('\n', ' ')
        texto_corto = (texto_completo[:150] + '...') if len(texto_completo) > 150 else texto_completo
        
        linea_ejemplo = f"   [Score: {fila['Seguridad_IA']:.4f}] {texto_corto}"
        print(linea_ejemplo)
        texto_reporte += linea_ejemplo + "\n"

# Guardar el archivo .txt
with open(Config.ARCHIVO_REPORTE, 'w', encoding='utf-8') as f:
    f.write(texto_reporte)

print("\n" + "="*60)
print(f"[EXITO] Analisis guardado en: {Config.DIR_ENTRADA_SALIDA}/")
print(f"Reporte de texto generado en: {Config.ARCHIVO_REPORTE}")
print("="*60)