#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# ==============================================================================
# 1. OBTENCIÓN Y ANÁLISIS DEL CVAR
# ==============================================================================

print("1. Cargando datos de riesgo...")
df_riesgo = pd.read_csv('perfil_riesgo_empresas.csv')

# Filtramos posibles nulos
df_riesgo = df_riesgo.dropna(subset=['CVaR_95'])

# Ordenamos de peor CVaR (más negativo) a mejor
df_sorted = df_riesgo.sort_values(by='CVaR_95', ascending=True)

print("2. Generando el gráfico de CVaR (95%)...")
plt.figure(figsize=(20, 8))

# Gráfico de barras
sns.barplot(
    data=df_sorted, 
    x='Ticker', 
    y='CVaR_95', 
    hue='indice', 
    dodge=False, 
    palette='viridis' # Cambiamos a viridis para variar el estilo visual
)

# Estética y etiquetas
plt.axhline(0, color='black', linewidth=1.5)
plt.title('Conditional Value at Risk (CVaR 95%) - Pérdida media en eventos de cola', fontsize=18, pad=20)
plt.ylabel('Pérdida Media Esperada (en el peor 5% de los casos)', fontsize=14)
plt.xlabel('Empresa', fontsize=14)

# Rotar tickers
plt.xticks(rotation=90, fontsize=10)

# Eje Y en porcentaje
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.legend(title='Índice de Referencia', loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

print("3. Guardando gráfico...")
nombre_archivo = 'comparativa_cvar_95.png'
plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
plt.show()

print(f" Gráfico guardado como '{nombre_archivo}'")