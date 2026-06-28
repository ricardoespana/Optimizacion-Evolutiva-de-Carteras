#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANALISIS DE SENTIMIENTO NLP - MODELO MULTILINGÜE (1 a 5 ESTRELLAS)
Toma las noticias financieras y las evalúa asignando una puntuación de estrellas.
Implementa aceleración por hardware (CUDA/MPS) y estructura de directorios profesional.
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import time
import os
import sys

# ==============================================================================
# 0. CONFIGURACIÓN GLOBAL Y DIRECTORIOS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    ARCHIVO_ENTRADA = os.path.join(BASE_DIR, 'dataset_2021_2026_eleconomista.csv')
    
    # Configuración del Modelo
    MODELO_HUGGINGFACE = "nlptown/bert-base-multilingual-uncased-sentiment"
    MODELO_NOMBRE_CORTO = "nlptown_bert"
    
    # Rutas de Salida
    DIR_SALIDA = os.path.join(BASE_DIR, "resultados_NLP", MODELO_NOMBRE_CORTO)
    ARCHIVO_SALIDA = os.path.join(DIR_SALIDA, f"{MODELO_NOMBRE_CORTO}_results.csv")

os.makedirs(Config.DIR_SALIDA, exist_ok=True)

# ==============================================================================
# 1. COMPROBACIÓN DEL ENTORNO Y HARDWARE
# ==============================================================================
tiempo_inicio = time.time()
print("⏳ Iniciando Análisis de Sentimiento (1 a 5 Estrellas)...")
print("="*60)
print(f"-> PyTorch versión: {torch.__version__}")

# Detección de hardware para máxima velocidad (CUDA para NVIDIA, MPS para Apple Silicon, CPU por defecto)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("-> 🚀 Hardware acelerador detectado: NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("-> 🚀 Hardware acelerador detectado: Apple Silicon (MPS)")
else:
    device = torch.device("cpu")
    print("-> ⚠️ No se detectó GPU. Ejecutando en CPU (modo lento).")
print("="*60)

# ==============================================================================
# 2. CARGA DE DATOS Y MODELO
# ==============================================================================
if not os.path.exists(Config.ARCHIVO_ENTRADA):
    print(f"❌ ERROR: No se encuentra el dataset: {Config.ARCHIVO_ENTRADA}")
    sys.exit(1)

df = pd.read_csv(Config.ARCHIVO_ENTRADA)
print(f"-> Dataset cargado: {len(df)} noticias listas para procesar.")

print(f"-> Descargando/Cargando modelo: {Config.MODELO_HUGGINGFACE}...")
tokenizer = AutoTokenizer.from_pretrained(Config.MODELO_HUGGINGFACE)
modelo = AutoModelForSequenceClassification.from_pretrained(Config.MODELO_HUGGINGFACE)

# Enviar el modelo a la tarjeta gráfica/acelerador y ponerlo en modo evaluación
modelo.to(device)
modelo.eval()

# ==============================================================================
# 3. PROCESAMIENTO NLP
# ==============================================================================
etiquetas_estrellas = []
scores_seguridad = []

for texto in tqdm(df['Texto_Analizar'], desc="Analizando noticias"):
    try:
        # Tokenización y envío a la GPU/MPS
        inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = modelo(**inputs)
            
        probabilidades = torch.nn.functional.softmax(outputs.logits, dim=-1)
        score_max, clase_idx = torch.max(probabilidades, dim=-1)
        
        # Etiqueta original ('1 star', '2 stars', etc.)
        label_cruda = modelo.config.id2label[clase_idx.item()]
        score = score_max.item()
        
        # Extracción del número entero
        estrella = int(label_cruda.split()[0])
        
        etiquetas_estrellas.append(estrella)
        scores_seguridad.append(score)
        
    except Exception as e:
        # Control de fallos silencioso para no romper el bucle
        etiquetas_estrellas.append(-1) 
        scores_seguridad.append(0.0)

# ==============================================================================
# 4. GUARDADO DE RESULTADOS
# ==============================================================================
df['Estrellas'] = etiquetas_estrellas
df['Seguridad_IA'] = scores_seguridad

df.to_csv(Config.ARCHIVO_SALIDA, index=False, encoding='utf-8-sig')

t_fin = time.time()
mins = int((t_fin - tiempo_inicio) // 60)
segs = int((t_fin - tiempo_inicio) % 60)

print("\n" + "="*60)
print(f"✅ ¡Análisis NLP Multilingüe completado!")
print(f"📁 Resultados guardados en: {Config.ARCHIVO_SALIDA}")
print(f"⏱️ Tiempo total de análisis NLP: {mins} minutos y {segs} segundos.")
print("="*60)