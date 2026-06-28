#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import os
import time

# ==============================================================================
# 0. INICIO DEL CRONÓMETRO Y CARPETAS
# ==============================================================================
tiempo_inicio = time.time()
print("Iniciando análisis de Riesgo-Rentabilidad y CML...")

# Creamos la carpeta específica para guardar este gráfico
dir_riesgo = "Analisis_Riesgo"
os.makedirs(dir_riesgo, exist_ok=True)
print(f"Directorio verificado/creado: ./{dir_riesgo}/")

# ==============================================================================
# 1. PARÁMETROS FINANCIEROS Y CARGA DE DATOS
# ==============================================================================
# Tasa Libre de Riesgo (Risk-Free Rate). Ajustable según el bono a 10 años (ej. 3.0%)
TASA_LIBRE_RIESGO = 0.02

print("1. Cargando el perfil de riesgo...")
# Cargamos los datos
df_riesgo = pd.read_csv('perfil_riesgo_empresas.csv')
# Eliminamos cualquier activo que no tenga datos válidos de rentabilidad o volatilidad
df_riesgo = df_riesgo.dropna(subset=['Rentabilidad_Anualizada', 'Volatilidad_Anualizada'])

# Comprobación de seguridad para el nombre de la columna (mayúscula/minúscula)
col_indice = 'Indice' if 'Indice' in df_riesgo.columns else 'indice'

# ==============================================================================
# 2. CÁLCULO DE LA LÍNEA DE MERCADO DE CAPITALES (CML) EMPÍRICA
# ==============================================================================
# Calculamos el Ratio de Sharpe empírico para encontrar la tangente"
df_riesgo['Sharpe'] = (df_riesgo['Rentabilidad_Anualizada'] - TASA_LIBRE_RIESGO) / df_riesgo['Volatilidad_Anualizada']

# Encontramos la acción con la mejor relación rentabilidad/riesgo
activo_estrella = df_riesgo.loc[df_riesgo['Sharpe'].idxmax()]
max_volatilidad_grafico = df_riesgo['Volatilidad_Anualizada'].max() * 1.1

# Puntos (X, Y) para trazar la CML: Desde Riesgo 0 hasta el final del gráfico
cml_x = np.array([0, max_volatilidad_grafico])
cml_y = TASA_LIBRE_RIESGO + activo_estrella['Sharpe'] * cml_x

# ==============================================================================
# 3. GENERACIÓN DEL GRÁFICO
# ==============================================================================
# --- CONFIGURACIÓN DE LA PALETA PERSONALIZADA (SET1) ---
# Extraemos los colores base de Set1: colores_base[0] es Rojo, colores_base[1] es Azul, colores_base[2] es Verde
colores_base = sns.color_palette('Set1', 3)

mapeo_colores = {
    'IBEX_SMALL_CAP': colores_base[0],   # Rojo
    'IBEX_MEDIUM_CAP': colores_base[1],  # Azul
    'IBEX 35': colores_base[2]           # Verde
    }

print("2. Generando el Mapa de Riesgo vs Rentabilidad con CML...")
plt.figure(figsize=(16, 10))

# Dispersión de activos coloreados por su índice
sns.scatterplot(
    data=df_riesgo,
    x='Volatilidad_Anualizada',
    y='Rentabilidad_Anualizada',
    hue=col_indice,
    s=100, alpha=0.8, palette= mapeo_colores
)

# Etiquetas de los Tickers junto a su punto en el scatterplot
for i in range(len(df_riesgo)):
    plt.text(
        x=df_riesgo['Volatilidad_Anualizada'].iloc[i] + 0.003,
        y=df_riesgo['Rentabilidad_Anualizada'].iloc[i] + 0.003,
        s=df_riesgo['Ticker'].iloc[i],
        fontsize=8, alpha=0.7
    )

# --- AÑADIDOS RELEVANTES: CML Y PUNTOS CLAVE ---
# 1. Dibujar la CML empírica
plt.plot(cml_x, cml_y, color='gold', linestyle='-', linewidth=2,
         label=f'Frontera Eficiente (Sharpe Max: {activo_estrella["Sharpe"]:.2f})')

# 2. Marcar la Tasa Libre de Riesgo (Risk-Free Rate) en el eje Y
plt.scatter(0, TASA_LIBRE_RIESGO, color='black', s=150, marker='*', zorder=5)
plt.text(0.005, TASA_LIBRE_RIESGO, 'Risk-Free (2%)', fontsize=10, fontweight='bold')

# 3. Resaltar la "Cartera Tangente" (El activo más eficiente que toca la CML)
plt.scatter(activo_estrella['Volatilidad_Anualizada'], activo_estrella['Rentabilidad_Anualizada'],
            color='gold', s=200, edgecolors='black', zorder=5)
plt.text(activo_estrella['Volatilidad_Anualizada'] - 0.02, activo_estrella['Rentabilidad_Anualizada'] + 0.015,
         f'Max Sharpe ({activo_estrella["Ticker"]})', fontsize=10, fontweight='bold', color='darkgoldenrod')
# -----------------------------------------------

# Líneas base en 0 y en la volatilidad media
vol_media = df_riesgo['Volatilidad_Anualizada'].mean()
plt.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
plt.axvline(vol_media, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Volatilidad Media')

# Título dinámico contando cuántos activos se están evaluando
num_activos = len(df_riesgo)
plt.title(f'Mapa de Riesgo vs Rentabilidad - Universo Global ({num_activos} activos)', fontsize=18, pad=20)

# Etiquetas de los ejes
plt.xlabel('Riesgo (Volatilidad Anualizada)', fontsize=12)
plt.ylabel('Rentabilidad Anualizada', fontsize=12)

# Formatear ejes como porcentajes (ej. 10% en lugar de 0.10)
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Ajuste de Leyenda y Ejes
plt.xlim(left=-0.01) # Dejamos margen negativo a la izquierda para que se vea el 0 de la CML
plt.legend(title='Leyenda', loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

print("3. Guardando gráfico...")
# Guardamos utilizando la ruta dinámica de la carpeta recién creada
ruta_salida = os.path.join(dir_riesgo, 'mapa_riesgo_rentabilidad_cml.png')
plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
plt.close() # Cierre de figura para no colapsar memoria

# ==============================================================================
# 4. FIN DEL CRONÓMETRO
# ==============================================================================
tiempo_fin = time.time()
tiempo_total = tiempo_fin - tiempo_inicio

horas = int(tiempo_total // 3600)
minutos = int((tiempo_total % 3600) // 60)
segundos = int(tiempo_total % 60)

print("-" * 50)
print(f"¡Gráfico generado y guardado exitosamente en ./{dir_riesgo}/!")
print(f"El script de Perfil de Riesgo ha tardado: {horas} horas, {minutos} minutos y {segundos} segundos.")
print("-" * 50)