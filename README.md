# Optimizacion-Evolutiva-de-Carteras
Repositorio de código utilizado para la elaboración del proyecto de fin de máster 
<!--
TFM_Optimizacion_Evolutiva_Carteras/
│
├── README.md                # El documento principal de presentación de tu proyecto
├── requirements.txt         # Lista de librerías de Python necesarias (Gymnasium, Pandas, PyTorch, etc.)
│
├── 1_Extraccion_Datos/      # Fase 1: Obtención y limpieza de datos
│   ├── get_market_data.py   # Scripts para descargar precios (LSEG/Refinitiv, Yahoo Finance, etc.)
│   ├── get_fundamentals.py  # Scripts para extraer variables contables
│   └── get_news_nlp.py      # Scripts de extracción de noticias para el análisis de sentimiento
│
├── 2_Procesamiento_Datos/   # Fase 2: Procesamiento y creación del Dataset
│   ├── data_cleaning.py     # Scripts para manejar valores nulos y normalización
│   ├── technical_indicators.py # Scripts para calcular RSI, MACD, Bollinger, etc.
│   ├── nlp_sentiment.py     # Scripts del modelo NLP (BERT/FinBERT) para puntuar noticias
│   └── merge_dataset.py     # Script que une todo en tu 'dataset_GA_estandarizado_v3.csv'
│
│
├── 3_Algoritmo_genetico/    # Fase 3: El núcleo de la Inteligencia Artificial
│   ├── genetic_mlp.py       # Definición de la red neuronal (GeneticMLPPolicy)
│   ├── ga_optimizer.py      # Lógica del Algoritmo Genético (Cruce, Mutación, Torneo)
│   └── train_ga_multistock_v2.py # Script principal de entrenamiento
│
├── 4_random_forest/      # Fase 4: Modelo sustituto
│   ├── backtest_results.py  # Scripts para calcular Sharpe, Drawdown, CAGR en test
│   └── xai_shap_analysis.py # Scripts del modelo Random Forest sustituto y gráficos SHAP
-->
