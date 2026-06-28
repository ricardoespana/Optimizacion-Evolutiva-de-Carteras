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
FECHA_INICIO = (2021, 1, 1)        # Año, Mes, Día (Inicio histórico TFM)
FECHA_FIN = (2026, 12, 31)         # Año, Mes, Día (Fin histórico TFM)

# --- NUEVOS PARÁMETROS DE FILTRADO ---
# Ejemplos: "eleconomista.es", "expansion.com", "cincodias.elpais.com"
DOMINIO_PERIODICO = "eleconomista.com" 

# Ajustamos los límites de búsqueda al tener un abanico temporal de 5 años
MAX_RESULTADOS_GOOGLE = 100        # Máximo permitido por Google por petición
NOTICIAS_OBJETIVO_POR_EMPRESA = 80 # Ampliamos el cupo a guardar por empresa

ARCHIVO_ENTRADA = 'df_u_lseg.csv'       # archivo con el universo de acciones
# Nombre dinámico para saber qué diario y fechas contiene
ARCHIVO_SALIDA = f'dataset_IBEX_2021_2026_{DOMINIO_PERIODICO.split(".")[0]}.csv' 
# ==============================================================================


def extraer_historico_diccionario(df_universo):
    """
    Función que recorre el universo de acciones, busca su histórico de noticias
    acotado a un diario específico, y almacena resultados por Ticker.
    """
    google_news = GNews(
        language='es', #IMPORTANTISIMO CAMBIAR PARA CADA PAIS
        country='ES', #IMPORTANTISIMO CAMBIAR PARA CADA PAIS
        max_results=MAX_RESULTADOS_GOOGLE,
        start_date=FECHA_INICIO,
        end_date=FECHA_FIN
    )
    
    diccionario_empresas = {}
    
    print(f"Iniciando extracción histórica ({FECHA_INICIO} a {FECHA_FIN})...")
    print(f"Filtro de fuente activa: {DOMINIO_PERIODICO.upper()}\n")

    for index, row in df_universo.iterrows():
        ticker = row['Ticker']
        nombre_empresa = row['Nombre']
        sector = row['Sector']
        indice = row['Indice']
        
        diccionario_empresas[ticker] = []
        
        print(f"[{index+1}/{len(df_universo)}] Buscando histórico: {nombre_empresa} ({ticker})...")
        
        # --- REEMPLAZA ESTE BLOQUE DENTRO DE TU FUNCIÓN ---
        
        # 1. Usar el Nombre de la empresa en lugar de solo el Ticker
        # 2. Eliminar la palabra "bolsa" que limita demasiado los resultados
        # 3. Quitar las comillas estrictas para que Google sea más flexible
        if DOMINIO_PERIODICO:
            query = f'{nombre_empresa} site:{DOMINIO_PERIODICO}'
        else:
            query = f'{nombre_empresa} financial news'
            
        try:
            # Añadimos un pequeño log para ver qué está buscando realmente el script
            print(f"      -> Query enviada: {query}")
            noticias = google_news.get_news(query)
            
            if not noticias:
                # Si falla con el nombre, probamos una segunda opción de emergencia: el Ticker
                print(f"      -> Sin éxito. Intentando búsqueda secundaria con Ticker: {ticker}")
                query_alt = f'{ticker} stock'
                noticias = google_news.get_news(query_alt)
                
            exitos = 0
            
            for noticia in noticias:
                if exitos >= NOTICIAS_OBJETIVO_POR_EMPRESA:
                    break 
                    
                fuente = noticia.get('publisher', {}).get('title', 'Desconocida')
                
                try:
                    articulo = google_news.get_full_article(noticia['url'])
                    
                    if articulo and len(articulo.text) > 200:
                        texto_final = articulo.text
                        tipo = "Texto Completo"
                    else:
                        texto_final = noticia.get('description', '')
                        tipo = "Solo Resumen"
                        if len(texto_final) < 50: 
                            continue 
                            
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
                    pass 
            
            print(f"  -> Guardadas {exitos} noticias en diccionario_empresas['{ticker}'].")
            
        except Exception as e:
            print(f"  -> [Error Servidor]: {e}")
            
        time.sleep(3) 
        
    print("\n" + "="*50 + "\n¡Extracción y volcado a memoria finalizado!\n" + "="*50)
    return diccionario_empresas

# ==============================================================================
# 2. EJECUCIÓN Y TRANSFORMACIÓN A DATAFRAME
# ==============================================================================

tiempo_inicio = time.time()
print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Arrancando pipeline de datos...")

df_u = pd.read_csv(ARCHIVO_ENTRADA)
datos_memoria = extraer_historico_diccionario(df_u)

lista_plana_noticias = []
for ticker, lista_noticias in datos_memoria.items():
    lista_plana_noticias.extend(lista_noticias) 

df_historico = pd.DataFrame(lista_plana_noticias)
df_historico.to_csv(ARCHIVO_SALIDA, index=False, encoding='utf-8-sig')

tiempo_fin = time.time()
tiempo_total_segundos = tiempo_fin - tiempo_inicio
tiempo_formateado = str(datetime.timedelta(seconds=int(tiempo_total_segundos)))

print(f"\nDatos guardados exitosamente en: {ARCHIVO_SALIDA}")
print(f"Total de noticias históricas consolidadas: {len(df_historico)}")
print(f"Tiempo total de ejecución (Auditoría): {tiempo_formateado}")