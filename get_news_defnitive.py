#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
PARA USAR ESTE SCRIPT DEBEREMOS UTILIZAR EL INTERPRETE DE PYTHON 3.13
DADO QUE EL PAQUETE GNEWS YA NO ESTA OPTIMIZADO PARA 3.11
https://github.com/ranahaani/GNews/blob/master/setup.py
"""

import pandas as pd
from gnews import GNews
import time
import datetime 
# ==============================================================================
# 1. PARÁMETROS DE CONFIGURACIÓN 
# ==============================================================================
FECHA_INICIO = (2025, 1, 1)        # Año, Mes, Día
FECHA_FIN = (2026, 3, 31)          # Año, Mes, Día
MAX_RESULTADOS_GOOGLE = 50         # Cuántas noticias le pedimos a Google por empresa
NOTICIAS_OBJETIVO_POR_EMPRESA = 10 # Cuántas noticias válidas queremos guardar como máximo
ARCHIVO_ENTRADA = 'df_u.csv'       # archivo con el universo de acciones
ARCHIVO_SALIDA = 'dataset_historico_25_26.csv' # Nombre del archivo final
# ==============================================================================


def extraer_historico_diccionario(df_universo):
    """
    Función que recorre el universo de acciones, busca su histórico de noticias
    y almacena los resultados en un diccionario estructurado por Ticker.
    """
    
    # Inicializamos el cliente de Google News con las fechas parametrizadas
    google_news = GNews(
        language='es', 
        country='ES', 
        max_results=MAX_RESULTADOS_GOOGLE,
        start_date=FECHA_INICIO,
        end_date=FECHA_FIN
    )
    
    # ----------------------------------------------------------------------
    # LA ESTRUCTURA DE DATOS: Un diccionario maestro
    # Formato: { 'SAN.MC': [ {noticia1}, {noticia2} ], 'BBVA.MC': [ ... ] }
    # ----------------------------------------------------------------------
    diccionario_empresas = {}
    
    print(f"Iniciando extracción histórica ({FECHA_INICIO} a {FECHA_FIN}) para {len(df_universo)} empresas...\n")

    # Bucle principal sobre el universo de acciones
    for index, row in df_universo.iterrows():
        ticker = row['Ticker']
        nombre_empresa = row['nombre']
        sector = row['sector']
        indice = row['indice']
        
        # Preparamos la clave del diccionario para esta empresa
        diccionario_empresas[ticker] = []
        
        print(f"[{index+1}/{len(df_universo)}] Buscando histórico: {nombre_empresa} ({ticker})...")
        
        # Consulta de búsqueda limpia
        query = f'"{nombre_empresa}" bolsa'
        
        try:
            # Petición a Google
            noticias = google_news.get_news(query)
            
            if not noticias:
                print("  -> Sin resultados en este marco temporal.")
                continue
                
            exitos = 0
            
            # Procesamos la lista de noticias obtenidas
            for noticia in noticias:
                if exitos >= NOTICIAS_OBJETIVO_POR_EMPRESA:
                    break # Salimos del bucle si ya tenemos las que queríamos
                    
                fuente = noticia.get('publisher', {}).get('title', 'Desconocida')
                
                try:
                    articulo = google_news.get_full_article(noticia['url'])
                    
                    # PLAN A: Extraer el texto completo
                    if articulo and len(articulo.text) > 200:
                        texto_final = articulo.text
                        tipo = "Texto Completo"
                        
                    # PLAN B: Usar el resumen de Google si hay paywall
                    else:
                        texto_final = noticia.get('description', '')
                        tipo = "Solo Resumen"
                        if len(texto_final) < 50: 
                            continue # Si tampoco hay resumen útil, la descartamos
                            
                    # Guardamos la noticia en la lista de la empresa correspondiente dentro del diccionario
                    diccionario_empresas[ticker].append({
                        'Ticker': ticker,
                        'Nombre': nombre_empresa,
                        'Sector': sector,
                        'Indice': indice,
                        'Fuente': fuente,
                        'Fecha_Publicacion': noticia.get('published date', ''),
                        'Titular': noticia.get('title', ''),
                        'Tipo_Extraccion': tipo,
                        'Texto_Analizar': texto_final
                    })
                    exitos += 1
                    
                except Exception:
                    pass # Ignora fallos puntuales de un link caído
            
            print(f"  -> Guardadas {exitos} noticias en diccionario_empresas['{ticker}'].")
            
        except Exception as e:
            print(f"  -> [Error Servidor]: {e}")
            
        # Pausa de cortesía para no saturar a Google (Vital al pedir tanto histórico)
        time.sleep(3) 
        
    print("\n" + "="*50 + "\n¡Extracción y volcado a memoria finalizado!\n" + "="*50)
    return diccionario_empresas

# ==============================================================================
# 2. EJECUCIÓN Y TRANSFORMACIÓN A DATAFRAME
# ==============================================================================

# --- INICIO DEL CRONÓMETRO ---
tiempo_inicio = time.time()
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Arrancando pipeline de datos...")

# 1. Cargamos el universo
df_u = pd.read_csv(ARCHIVO_ENTRADA)

# 2. Obtenemos el diccionario maestro lleno de datos
# (Esto se quedará guardado en tu Variable Explorer de Python)
datos_memoria = extraer_historico_diccionario(df_u)

# 3. Transformación del Diccionario a DataFrame (Aplanado)
# Convertimos el diccionario de listas en una única lista plana para Pandas
lista_plana_noticias = []
for ticker, lista_noticias in datos_memoria.items():
    lista_plana_noticias.extend(lista_noticias) # Extend junta las listas

# 4. Creamos el DataFrame final y lo exportamos a CSV
df_historico = pd.DataFrame(lista_plana_noticias)
df_historico.to_csv(ARCHIVO_SALIDA, index=False, encoding='utf-8-sig')

# --- FIN DEL CRONÓMETRO ---
tiempo_fin = time.time()
tiempo_total_segundos = tiempo_fin - tiempo_inicio

# Convertimos los segundos puros a formato HH:MM:SS
tiempo_formateado = str(datetime.timedelta(seconds=int(tiempo_total_segundos)))
# Reporte final
print(f"\nDatos guardados exitosamente en: {ARCHIVO_SALIDA}")
print(f"Total de noticias históricas consolidadas: {len(df_historico)}")
print(f"Tiempo total de ejecución (Auditoría): {tiempo_formateado}")


"""
Datos guardados exitosamente en: dataset_historico_25_26.csv
Total de noticias históricas consolidadas: 832
Tiempo total de ejecución (Auditoría): 0:15:47
"""