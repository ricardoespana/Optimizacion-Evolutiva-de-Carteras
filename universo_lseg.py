#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import lseg.data as ld
import pandas as pd
import time

# ==============================================================================
# INICIO DEL CRONÓMETRO
# ==============================================================================
tiempo_inicio = time.time()

# ==============================================================================
# PASO 1: CONEXIÓN
# ==============================================================================
ld.open_session()

# ==============================================================================
# PASO 2: DEFINICIÓN DE LO QUE VAMOS A BUSCAR
# ==============================================================================
#DEFINIENDO EL ÍNDICE CON 0# DA LA COMPOSICIÓN DEL ÍNDICE
# Esto le dice a LSEG: "No me des el índice, dame la lista de acciones que lo componen"
indices = {
#    "IBEX 35": "0#.IBEX",
#    "IBEX_MEDIUM_CAP": "0#.IBEXC",
 #   "IBEX_SMALL_CAP": "0#.IBEXS"
 "S&P 500": "0#.SPX"
}

campos_a_extraer = [
    "TR.ExchangeTicker",      
    "TR.CommonName",          
    "TR.TRBCEconomicSector",  
    "TR.TRBCIndustry"         
]

lista_de_tablas = []

# ==============================================================================
# PASO 3: DESCARGA DE LAS ACCIONES (CON REINTENTOS)
# ==============================================================================
for nombre_indice, codigo_ric in indices.items():
    print(f"\nExtrayendo las acciones que componen el: {nombre_indice}...")
    
    intentos_maximos = 3
    exito = False
    
    for intento in range(intentos_maximos):
        try:
            df_temporal = ld.get_data(
                universe=[codigo_ric],
                fields=campos_a_extraer
            )
            
            # En esta tabla que nos devuelve, la primera columna (Instrument) es el RIC de la acción
            df_temporal.columns = ["RIC", "Ticker", "Nombre", "Sector", "Industria"]
            df_temporal["Indice"] = nombre_indice
            lista_de_tablas.append(df_temporal)
            
            # Muestra cuántas acciones ha encontrado en ese índice
            numero_de_acciones = len(df_temporal)
            exito = True
            print(f"  -> ¡Éxito! Se han encontrado {numero_de_acciones} empresas en este índice.")
            
            time.sleep(2) 
            break  
            
        except Exception as e:
            print(f"  -> Fallo en el intento {intento + 1} de {intentos_maximos}. Esperando 5 segundos para reintentar...")
            time.sleep(5) 
            
    if not exito:
        print(f"  -> ERROR DEFINITIVO: No se pudo descargar {nombre_indice}. Saltando al siguiente...")

# ==============================================================================
# PASO 4: UNIÓN Y ORGANIZACIÓN DE LOS DATOS
# ==============================================================================
if len(lista_de_tablas) > 0:
    # Juntamos todas las empresas en una sola tabla
    df_final = pd.concat(lista_de_tablas, ignore_index=True)

    orden_columnas = ["RIC", "Ticker", "Nombre", "Sector", "Industria", "Indice"]
    df_final = df_final[orden_columnas]

    # ==============================================================================
    # PASO 5: EXPORTACIÓN
    # ==============================================================================
    nombre_archivo = "df_S&P.csv"
    df_final.to_csv(nombre_archivo, index=False, encoding="utf-8-sig")

    print(f"\n¡Proceso terminado! Tu universo de {len(df_final)} empresas se ha guardado como '{nombre_archivo}'.")
else:
    print("\nNo se ha podido generar el archivo porque fallaron todas las descargas.")
    
    
# ==============================================================================
#  DEFINICIÓN DE RICS PARA BENCHMARK REAL
# ==============================================================================
# CON ESTO OBTENEMOS LOS ÍNDICES
#rics_indices_oficiales = [".IBEX", ".IBEXC", ".IBEXS"] 
# .IBEX  -> IBEX 35
# .IBEXC -> IBEX Medium Cap
# .IBEXS -> IBEX Small Cap

#print(f"\nRICs oficiales de índices preparados para descarga: {rics_indices_oficiales}")


rics_indices_oficiales = ["SPX"] 
print(f"\nRICs oficiales de índices preparados para descarga: {rics_indices_oficiales}")

""" PARA EL CASO DEL S&P ES ASÍ
# 1. Tu lista original
lista_indice = ["SPX"]

# 2. Convertir la lista a un DataFrame de Pandas
df_indice_SP = pd.DataFrame(lista_indice, columns=['Indice'])

# 3. Ahora sí puedes exportar a CSV
df_indice_SP.to_csv("df_indice_S&P.csv", index=False)
"""
# ==============================================================================
# PASO 6: DESCONEXIÓN Y CRONÓMETRO
# ==============================================================================
ld.close_session()

tiempo_fin = time.time()
tiempo_total_segundos = tiempo_fin - tiempo_inicio

horas = int(tiempo_total_segundos // 3600)
minutos = int((tiempo_total_segundos % 3600) // 60)
segundos = int(tiempo_total_segundos % 60)

print(f"⏱️ Auditoría de tiempo: El script ha tardado {horas}h, {minutos}m y {segundos}s.")

import pandas as pd

