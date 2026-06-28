# -*- coding: utf-8 -*-
"""
Script definitivo para observar arbol representativo del Random Forest (.joblib)
y extraer tanto la ramificación visual como los pesos de las variables.
- Rutas dinámicas y creación automática de directorios.
- Extracción dinámica de variables para evitar ValueError.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from pathlib import Path

# ==============================================================================
# 1. PARÁMETROS Y CONFIGURACIÓN DE DIRECTORIOS
# ==============================================================================
ID_EXPERIMENTO = "GA_MultiStock_v28"

BASE_DIR = Path("./GA_Env") 

# Uso dinámico de ID_EXPERIMENTO para no tener que cambiar la ruta a mano
RUTA_MODELO = BASE_DIR / f"experimentos/{ID_EXPERIMENTO}/rf_homogeneo"
ARCHIVO_ENTRADA = RUTA_MODELO / "best_rf_model.joblib"

# Definimos y CREAMOS el directorio de salida si no existe
DIR_SALIDA = RUTA_MODELO / "Comparativas"
DIR_SALIDA.mkdir(parents=True, exist_ok=True) # <-- Vital para que no dé error al guardar

RUTA_SALIDA_ARBOL = DIR_SALIDA / "rf_ramificacion_arbol.png"
RUTA_SALIDA_PESOS = DIR_SALIDA / "rf_importancia_variables.png"

# ==============================================================================
# 2. DESPERTAR EL CEREBRO Y EXTRAER VARIABLES DINÁMICAMENTE
# ==============================================================================
print("Cargando el cerebro del Random Forest...")
rf_model = joblib.load(ARCHIVO_ENTRADA)

# Extraemos las variables directamente de la memoria del modelo
importancias = rf_model.feature_importances_

if hasattr(rf_model, 'feature_names_in_'):
    nombres_reales = rf_model.feature_names_in_
else:
    print(f"Aviso: El modelo no guardó los nombres. Se usarán etiquetas genéricas.")
    nombres_reales = [f"Variable_{i}" for i in range(len(importancias))]

# ==============================================================================
# 3. LA "FOTOGRAFÍA": RAMIFICACIÓN DE UN ÁRBOL INDIVIDUAL
# ==============================================================================
print("Generando diagrama de ramificación del Árbol 50...")

# Extraemos el árbol 50 del bosque
arbol_50 = rf_model.estimators_[50]

plt.style.use('seaborn-v0_8-white')
fig_tree, ax_tree = plt.subplots(figsize=(24, 12), facecolor='white')

# Dibujamos el árbol usando los nombres dinámicos
plot_tree(arbol_50, 
          feature_names=nombres_reales, 
          filled=True, 
          rounded=True, 
          max_depth=3, 
          fontsize=16, 
          ax=ax_tree)

plt.title("Estructura de Decisión (Árbol Representativo del Random Forest)", fontsize=30, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(RUTA_SALIDA_ARBOL, dpi=300, bbox_inches="tight", facecolor='white')
plt.close()

# ==============================================================================
# 4. LA VERDAD MATEMÁTICA: PONDERACIÓN GLOBAL DE LAS VARIABLES
# ==============================================================================
print("Calculando ponderación global (Feature Importances)...")

# Lo metemos en un DataFrame usando los nombres dinámicos
df_pesos = pd.DataFrame({
    'Variable': nombres_reales,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=True)

# Ajustamos la altura dinámicamente según la cantidad real de variables
altura_grafico = max(6, len(nombres_reales) * 0.4)
fig_pesos, ax_pesos = plt.subplots(figsize=(10, altura_grafico), facecolor='white')

# Gráfico de barras horizontales
bars = ax_pesos.barh(df_pesos['Variable'], df_pesos['Importancia'] * 100, color="#4c72b0", edgecolor="black")

for bar in bars:
    width = bar.get_width()
    ax_pesos.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                  f"{width:.1f}%", va="center", fontsize=10)

ax_pesos.set_title("Importancia Global de las Variables en el Random Forest", fontsize=14, fontweight='bold', pad=15)
ax_pesos.set_xlabel("Peso de decisión (%)", fontsize=11)
ax_pesos.grid(axis='x', linestyle='--', alpha=0.7)

ax_pesos.set_xlim(0, df_pesos['Importancia'].max() * 100 * 1.15)

plt.tight_layout()
plt.savefig(RUTA_SALIDA_PESOS, dpi=300, bbox_inches="tight", facecolor='white')
plt.close()

print(f"¡Listo! Imágenes extraídas y guardadas con éxito. Se detectaron {len(nombres_reales)} variables.")