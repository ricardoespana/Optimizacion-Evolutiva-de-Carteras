#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# ==============================================================================
# 1. GRAFICO DE DRAWDOWNS
# ==============================================================================

print("1. Cargando datos de riesgo...")
df_riesgo = pd.read_csv('perfil_riesgo_empresas.csv')

# Filtramos valores nulos por seguridad
df_riesgo = df_riesgo.dropna(subset=['Max_Drawdown'])

# Ordenamos el DataFrame de peor a mejor Max Drawdown (el más negativo primero)
df_sorted = df_riesgo.sort_values(by='Max_Drawdown', ascending=True)

print("2. Generando el gráfico de barras...")
# Hacemos la figura bastante ancha para que quepan bien los 56 tickers en el eje X
plt.figure(figsize=(20, 8))

# Creamos el gráfico de barras con Seaborn
# dodge=False evita que las barras se separen extrañamente al usar 'hue'
barplot = sns.barplot(
    data=df_sorted, 
    x='Ticker', 
    y='Max_Drawdown', 
    hue='indice', 
    dodge=False, 
    palette='Set1'
)

# Línea en el 0 para marcar la base
plt.axhline(0, color='black', linewidth=1.5)

# Formato y estilo
plt.title('Comparativa de Maximum Drawdown (Peor caída histórica)', fontsize=18, pad=20)
plt.ylabel('Maximum Drawdown (Caída desde máximo)', fontsize=14)
plt.xlabel('Empresa', fontsize=14)

# Rotamos las etiquetas del eje X a 90 grados para que se lean los tickers sin pisarse
plt.xticks(rotation=90, fontsize=10)

# Convertir el eje Y a formato porcentaje
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Ajustamos la leyenda para que no tape las barras (la ponemos abajo a la derecha)
plt.legend(title='Índice de Referencia', loc='lower right', fontsize=10)

# Añadimos una cuadrícula horizontal para facilitar la lectura de los porcentajes
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

print("3. Guardando el gráfico...")
nombre_archivo = 'comparativa_max_drawdown.png'
plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
plt.show()

print(f" Gráfico generado con éxito y guardado como '{nombre_archivo}'")