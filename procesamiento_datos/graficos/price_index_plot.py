#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JUSTIFICACIÓN METODOLÓGICA: NORMALIZACIÓN A BASE 100
En este bloque normalizamos los precios de cotización a "Base 100" para poder comparar el rendimiento 
relativo de distintas acciones de forma equitativa y visualmente clara. Si graficáramos los precios de 
cierre brutos, las acciones con un valor nominal muy alto aplastarían en el eje Y a aquellas con un 
valor nominal bajo, distorsionando la percepción de su evolución. Al reescalar todas las series temporales 
para que comiencen exactamente en el valor 100 el primer día de cotización válido, transformamos el 
gráfico en una representación directa de la rentabilidad acumulada. De este modo, cualquier variación 
se lee en términos porcentuales (ej. un valor final de 150 indica una revalorización exacta del 50%), 
independientemente de si la acción costaba originalmente 2€ o 200€.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# ==============================================================================
# 0. INICIO DEL CRONÓMETRO Y CARPETAS
# ==============================================================================
tiempo_inicio = time.time()
print("⏳ Iniciando normalización a Base 100 y generación de gráficos...")

# Crear el directorio principal para guardar los gráficos
dir_base100 = "Evolucion_Base100"
os.makedirs(dir_base100, exist_ok=True)
print(f"Directorio verificado/creado: ./{dir_base100}/")

# ==============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================
print("1. Cargando datos...")
df = pd.read_csv('cotizaciones_limpias.csv', parse_dates=['Date'])

# Comprobación de seguridad para el nombre de la columna (mayúscula/minúscula)
col_indice = 'Indice' if 'Indice' in df.columns else 'indice'

# Crear tabla pivote: Filas = Fechas, Columnas = Tickers, Valores = Precio de Cierre
df_close = df.pivot(index='Date', columns='Ticker', values='Close')

# Obtener los índices únicos
df_indices = df[['Ticker', col_indice]].drop_duplicates()
indices_unicos = df_indices[col_indice].dropna().unique()

# ==============================================================================
# 2. GENERACIÓN DE GRÁFICOS POR ÍNDICE (CONSTITUYENTES)
# ==============================================================================
print("2. Generando gráficos en Base 100 por Índice...")

# DataFrame vacío para ir guardando el comportamiento medio de cada índice (Estudio Macro)
df_macro_indices = pd.DataFrame(index=df_close.index)

for indice in indices_unicos:
    print(f"   -> Procesando gráfico para: {indice}...")
    
    # Seleccionar solo los tickers de este índice
    tickers = df_indices[df_indices[col_indice] == indice]['Ticker'].tolist()
    
    # Filtro de seguridad: nos aseguramos de que el ticker existe en la tabla pivote
    tickers_validos = [t for t in tickers if t in df_close.columns]
    precios_indice = df_close[tickers_validos]
    
    # NORMALIZACIÓN BASE 100: 
    # Dividimos todos los precios por el precio del primer día válido y multiplicamos por 100
    # Usamos bfill() por si alguna acción empezó a cotizar unos días más tarde
    precios_base_100 = (precios_indice / precios_indice.bfill().iloc[0]) * 100
    
    # Guardamos la media del índice para el gráfico MACRO
    df_macro_indices[indice] = precios_base_100.mean(axis=1)

    # Configurar el lienzo
    plt.figure(figsize=(14, 8))
    
    # Dibujar cada línea
    for col in precios_base_100.columns:
        plt.plot(precios_base_100.index, precios_base_100[col], label=col, alpha=0.8, linewidth=1.5)
    
    # Línea horizontal en 100 para marcar el punto de equilibrio (break-even)
    plt.axhline(100, color='black', linestyle='--', alpha=0.5, linewidth=2)
    
    # Estilos y etiquetas
    plt.title(f'Evolución del Precio (Base 100) - Constituyentes del {indice}', fontsize=16, pad=15)
    plt.ylabel('Valor (Base 100 = Inicio)', fontsize=12)
    plt.xlabel('Fecha', fontsize=12)
    
    # Leyenda fuera del gráfico para que no tape las líneas
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Guardar la imagen en su carpeta
    nombre_archivo = os.path.join(dir_base100, f"base100_acciones_{indice.replace(' ', '_')}.png")
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.close() # <- Cambio clave: Cerramos en lugar de plt.show() para proteger la memoria

# ==============================================================================
# 3. ESTUDIO MACRO: COMPARATIVA GLOBAL DE LOS ÍNDICES
# ==============================================================================
print("3. Generando gráfico MACRO (Comparativa media entre Índices)...")

plt.figure(figsize=(14, 8))

# Dibujar la línea de la media de cada índice
for col in df_macro_indices.columns:
    plt.plot(df_macro_indices.index, df_macro_indices[col], label=col, linewidth=2.5)

plt.axhline(100, color='black', linestyle='--', alpha=0.5, linewidth=2)

plt.title('Comparativa MACRO: Rendimiento Medio por Índice (Base 100)', fontsize=16, pad=15)
plt.ylabel('Valor Medio (Base 100 = Inicio)', fontsize=12)
plt.xlabel('Fecha', fontsize=12)

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

ruta_macro = os.path.join(dir_base100, "base100_MACRO_Global.png")
plt.savefig(ruta_macro, dpi=300, bbox_inches='tight')
plt.close()

print(f" Gráficos generados y guardados en el directorio '{dir_base100}'.")

# ==============================================================================
# 4. FIN DEL CRONÓMETRO
# ==============================================================================
tiempo_fin = time.time()
tiempo_total = tiempo_fin - tiempo_inicio

horas = int(tiempo_total // 3600)
minutos = int((tiempo_total % 3600) // 60)
segundos = int(tiempo_total % 60)

print("-" * 50)
print(f"El script Base 100 ha tardado: {horas} horas, {minutos} minutos y {segundos} segundos.")
print("-" * 50)