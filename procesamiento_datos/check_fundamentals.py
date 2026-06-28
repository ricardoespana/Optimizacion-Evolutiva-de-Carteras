#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUDITORÍA AVANZADA DE DATOS FUNDAMENTALES (2020-2025) CON GENERACIÓN DE REPORTE TXT
Este script verifica la calidad, consistencia temporal (24 trimestres estrictos)
y lógica financiera del panel limpio antes de ser inyectado en el entorno de trading.
Vuelca los resultados en un archivo de texto estructurado para su posterior lectura.
"""

import pandas as pd
import numpy as np
import os

# ==============================================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_PANEL = os.path.join(BASE_DIR, "Analisis_Fundamental/S&P_v2", "fundamentales_limpios.csv")
RUTA_REPORTE_TXT = os.path.join(BASE_DIR, "Analisis_Fundamental/S&P_v2", "Auditoria_Fundamentales_v2.txt")

def auditar_panel(ruta_archivo, ruta_salida_txt):
    if not os.path.exists(ruta_archivo):
        print(f"❌ ERROR: No se encuentra el archivo en {ruta_archivo}")
        return

    # Cargar datos
    df = pd.read_csv(ruta_archivo)
    
    # Identificar la columna de activos utilizada (Ticker, Instrument o RIC)
    id_col = 'Ticker' if 'Ticker' in df.columns else ('Instrument' if 'Instrument' in df.columns else 'RIC')
    
    # Buffer para almacenar las líneas de texto del reporte
    lineas_reporte = []

    def escribir_linea(texto=""):
        """Imprime en consola y almacena en el buffer del archivo txt"""
        print(texto)
        lineas_reporte.append(texto + "\n")

    # ==============================================================================
    # MAQUETACIÓN DEL INFORME EN EL TXT
    # ==============================================================================
    escribir_linea("=" * 70)
    escribir_linea("🔍 INFORME DE AUDITORÍA DE DATOS FUNDAMENTALES (RANGO 2020-2025)")
    escribir_linea("=" * 70)

    # 1. RESUMEN BÁSICO
    total_filas = len(df)
    total_activos = df[id_col].nunique()
    escribir_linea("\n[1] RESUMEN GENERAL DEL DATASET:")
    escribir_linea(f"    -> Total de registros en el panel (Filas): {total_filas}")
    escribir_linea(f"    -> Activos únicos detectados ({id_col}): {total_activos}")
    escribir_linea(f"    -> Rango temporal detectado: {df['Anio'].min()} - {df['Anio'].max()}")

    # 2. AUDITORÍA DE VALORES NULOS (NaN)
    escribir_linea("\n[2] DETECCIÓN DE VALORES NULOS POR VARIABLE:")
    nulos = df.isnull().sum()
    porcentaje_nulos = (nulos / total_filas) * 100
    
    df_nulos = pd.DataFrame({'Nulos': nulos, 'Porcentaje (%)': porcentaje_nulos})
    df_nulos = df_nulos[df_nulos['Nulos'] > 0].sort_values(by='Porcentaje (%)', ascending=False)
    
    if df_nulos.empty:
        escribir_linea("    ✅ ¡Excelente! No hay valores nulos en ninguna variable del panel.")
    else:
        escribir_linea("    (Nota: Es normal encontrar nulos en métricas sectoriales, ej: EBITDA en Banca)")
        for index, row in df_nulos.iterrows():
            escribir_linea(f"    -> {index.ljust(25)}: {str(int(row['Nulos'])).rjust(6)} nulos ({row['Porcentaje (%)']:.2f}%)")

    # 3. CONSISTENCIA TEMPORAL ESTRICTA (24 Trimestres entre 2020 y 2025)
    escribir_linea("\n[3] CONSISTENCIA CRONOLÓGICA ESTRICTA (Ventanas temporales por activo):")
    
    # Identificador único de periodo trimestral por fila
    df['_periodo_str'] = df['Anio'].astype(str) + "_Q" + df['Trimestre'].astype(str)
    trimestres_totales_esperados = 24  # 4 trimestres * 6 años (2020-2025)
    
    conteo_trimestres = df.groupby(id_col)['_periodo_str'].nunique()
    activos_incompletos = conteo_trimestres[conteo_trimestres < trimestres_totales_esperados]
    
    if activos_incompletos.empty:
        escribir_linea(f"    ✅ Estructuralmente perfecto: Todos los {total_activos} activos cuentan con los {trimestres_totales_esperados} trimestres secuenciales completos (2020 Q1 - 2025 Q4).")
    else:
        escribir_linea(f"    ⚠️ {len(activos_incompletos)} activos presentan series incompletas (< {trimestres_totales_esperados} trimestres).")
        escribir_linea("    [DETALLE DE ACTIVOS AFECTADOS]:")
        for act, num in activos_incompletos.items():
            escribir_linea(f"       -> {act.ljust(10)}: Ofrece {str(num).rjust(2)} de {trimestres_totales_esperados} trimestres.")

    # 4. AUDITORÍA DE GRUPOS CONTABLES
    escribir_linea("\n[4] DESGLOSE DEL UNIVERSO POR MODELO CONTABLE:")
    if 'Grupo_Contable' in df.columns:
        desglose = df.groupby('Grupo_Contable')[id_col].nunique()
        for grupo, conteo in desglose.items():
            escribir_linea(f"    -> {grupo.ljust(15)}: {str(conteo).rjust(3)} activos únicos")
    else:
        escribir_linea("    ❌ CRÍTICO: La columna 'Grupo_Contable' no se encuentra en el archivo.")

    # 5. CONTROL DE ANOMALÍAS FINANCIERAS
    escribir_linea("\n[5] FILTRO DE ANOMALÍAS Y LÓGICA FINANCIERA:")
    alertas = 0
    
    if 'Capitalizacion_Bursatil' in df.columns:
        cap_cero_neg = len(df[df['Capitalizacion_Bursatil'] <= 0])
        if cap_cero_neg > 0:
            escribir_linea(f"    ❌ Alerta: Hay {cap_cero_neg} registros con Capitalización Bursátil <= 0.")
            alertas += 1

    if 'PER' in df.columns:
        # Alertas de múltiplos de valoración extremos que desestabilizan redes PPO
        per_anormal = len(df[(df['PER'] > 500) | (df['PER'] < -100)])
        if per_anormal > 0:
            escribir_linea(f"    ⚠️ Advertencia: Detectados {per_anormal} registros con múltiplos PER extremos (< -100 o > 500).")
            alertas += 1

    if 'Rentabilidad_Dividendo' in df.columns:
        div_anormal = len(df[df['Rentabilidad_Dividendo'] > 40])
        if div_anormal > 0:
            escribir_linea(f"    ⚠️ Advertencia: Hay {div_anormal} registros con Dividend Yield > 40% (Outliers).")
            alertas += 1

    if alertas == 0:
        escribir_linea("    ✅ Limpieza financiera superada: No se detectan anomalías extremas en múltiplos contables.")

    escribir_linea("\n" + "=" * 70)
    escribir_linea("FIN DEL INFORME - DATASET APTO PARA REDES NEURONALES / ENTORNO RL")
    escribir_linea("=" * 70)

    # ==============================================================================
    # VOLCADO FINAL AL ARCHIVO .TXT
    # ==============================================================================
    with open(ruta_salida_txt, "w", encoding="utf-8") as archivo_txt:
        archivo_txt.writelines(lineas_reporte)
    
    print(f"\n💾 Reporte escrito con éxito para su lectura en: {ruta_salida_txt}")

if __name__ == "__main__":
    auditar_panel(ARCHIVO_PANEL, RUTA_REPORTE_TXT)