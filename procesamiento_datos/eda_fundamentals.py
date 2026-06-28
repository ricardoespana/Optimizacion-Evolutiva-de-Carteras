"""
ANÁLISIS EXPLORATORIO DE DATOS (EDA) - FUNDAMENTALES & TÉCNICOS
Genera un diagnóstico completo del panel de fundamentales:
1. Análisis de completitud (Datos Faltantes por Grupo y Variable).
2. Estadísticos descriptivos y Boxplots.
3. Correlación de variables segmentada por modelo contable.
4. Correlación cruzada (Cross-Correlation) entre Fundamentales e Indicadores Técnicos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. CONFIGURACIÓN GLOBAL
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    ARCHIVO_ENTRADA = os.path.join(BASE_DIR, "Analisis_Fundamental/S&P_v2", "fundamentales_limpios.csv")
    ARCHIVO_TECNICOS = os.path.join(BASE_DIR, "cotiz_SP_technical_completo.csv") 
    
    DIR_SALIDA = os.path.join(BASE_DIR, "reporte_eda", "fundamentales_S&P_v2")
    DIR_SALIDA_FALTANTES = os.path.join(DIR_SALIDA, "datos_faltantes") # Nueva ruta principal
    DIR_SALIDA_CROSS = os.path.join(DIR_SALIDA, "Correlaciones_Tecnicas")
    
    COLS_DESCRIPTIVAS = ['Date', 'Anio', 'Trimestre', 'RIC', 'Ticker', 'Nombre', 'Indice', 'TRBC_Sector', 'Grupo_Contable', 'Instrument']

os.makedirs(Config.DIR_SALIDA, exist_ok=True)
os.makedirs(Config.DIR_SALIDA_FALTANTES, exist_ok=True)
os.makedirs(Config.DIR_SALIDA_CROSS, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper")

# ==============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS FUNDAMENTALES
# ==============================================================================
tiempo_inicio = time.time()
print("⏳ Iniciando EDA Profesional de Fundamentales y Técnicos...")

if not os.path.exists(Config.ARCHIVO_ENTRADA):
    print(f"❌ ERROR: No se encuentra el archivo {Config.ARCHIVO_ENTRADA}")
    sys.exit(1)

df = pd.read_csv(Config.ARCHIVO_ENTRADA)
df['Date'] = pd.to_datetime(df['Date'])

if 'Ticker' in df.columns:
    df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()

df['Grupo_Contable'] = df['Grupo_Contable'].fillna('SIN_CLASIFICAR')
df_limpio = df.loc[:, ~df.columns.str.contains(r'\.\d+')]
if 'Fecha' in df_limpio.columns:
    df_limpio = df_limpio.drop(columns=['Fecha'])

print("-> Saneando tipos de datos fundamentales...")
cols_a_convertir = [c for c in df_limpio.columns if c not in Config.COLS_DESCRIPTIVAS]
for c in cols_a_convertir:
    df_limpio[c] = pd.to_numeric(df_limpio[c], errors='coerce')

cols_financieras = [c for c in cols_a_convertir if df_limpio[c].notna().sum() > 0]
print(f"-> Dataset Fundamental listo: {df_limpio.shape[0]} registros y {len(cols_financieras)} métricas útiles.")


# ==============================================================================
# 2. ANÁLISIS DE COMPLETITUD (HEATMAPS DE DISPONIBILIDAD POR GRUPO Y VARIABLE)
# ==============================================================================
print("\n1. Generando Heatmaps de Disponibilidad (Iterando por Grupos y Variables)...")

for grupo in df_limpio['Grupo_Contable'].unique():
    if grupo == 'SIN_CLASIFICAR': continue
    
    # Crear subcarpeta para el grupo contable actual
    dir_grupo = os.path.join(Config.DIR_SALIDA_FALTANTES, str(grupo))
    os.makedirs(dir_grupo, exist_ok=True)
    
    # Aislar datos de este grupo
    df_grupo = df_limpio[df_limpio['Grupo_Contable'] == grupo]
    
    # Filtrar solo variables que tengan algún dato en este grupo específico
    cols_grupo = [c for c in cols_financieras if df_grupo[c].notna().sum() > 0]
    
    print(f"   -> Procesando {grupo}: {len(cols_grupo)} variables a mapear.")
    
    for var in cols_grupo:
        # 1. Preparar la tabla (Eliminamos duplicados preventivamente para que .pivot no falle)
        df_unico = df_grupo.drop_duplicates(subset=['Date', 'Ticker'])
        pivot_availability = df_unico.pivot(index='Date', columns='Ticker', values=var)

        # 2. Conversión forzosa y ordenación temporal
        pivot_availability.index = pd.to_datetime(pivot_availability.index, errors='coerce')
        pivot_availability = pivot_availability[pivot_availability.index.notna()]
        pivot_availability = pivot_availability.sort_index()

        # 3. Cálculo de los puntos de cambio de año para las etiquetas (Eje Y)
        anios = pivot_availability.index.year
        cambios_anio = np.where(anios != pd.Series(anios).shift(1))[0]
        etiquetas_anio = anios[cambios_anio].astype(int)

        # 4. Generación del gráfico
        plt.figure(figsize=(16, 8))
        ax = sns.heatmap(pivot_availability.isnull(), cbar=False, yticklabels=False, cmap='viridis')

        # 5. Aplicar las etiquetas anuales al eje Y
        ax.set_yticks(cambios_anio)
        ax.set_yticklabels(etiquetas_anio, rotation=0, fontsize=10)

        plt.title(f'[{grupo}] Disponibilidad: {var} (Amarillo = Faltantes)', fontsize=14, fontweight='bold')
        plt.xlabel('Empresas (Tickers)', fontsize=12)
        plt.ylabel('Año', fontsize=12)
        plt.tight_layout()

        # 6. Guardado
        # Reemplazamos barras en el nombre por si alguna variable tiene '/' (ej. P/E)
        nombre_archivo = f"heatmap_faltantes_{var}.png".replace("/", "_")
        plt.savefig(os.path.join(dir_grupo, nombre_archivo), dpi=300)
        plt.close()


# ==============================================================================
# 3. ESTADÍSTICOS DESCRIPTIVOS Y OUTLIERS
# ==============================================================================
print("\n2. Generando Estadísticos Globales y Boxplots...")

df_desc = df_limpio[cols_financieras].dropna(axis=1, how='all').describe().T
df_desc.to_csv(os.path.join(Config.DIR_SALIDA, "Estadisticos_Descriptivos_Globales.csv"))

metricas_box = ['PER', 'ROA_Trimestral', 'Rentabilidad_Dividendo', 'Precio_Valor_Contable']
metricas_box = [m for m in metricas_box if m in df_limpio.columns]

for metrica in metricas_box:
    if df_limpio[metrica].notna().sum() == 0: continue
    fig, ax = plt.subplots(figsize=(10, 5))
    p1 = df_limpio[metrica].quantile(0.01)
    p99 = df_limpio[metrica].quantile(0.99)
    df_plot = df_limpio[(df_limpio[metrica] >= p1) & (df_limpio[metrica] <= p99)]
    sns.boxplot(data=df_plot, x='Grupo_Contable', y=metrica, palette='Set2', ax=ax)
    ax.set_title(f'Distribución de {metrica} por Grupo Contable (Filtro Outliers 1%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.DIR_SALIDA, f"02_Boxplot_{metrica}.png"), dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# 4. ANÁLISIS DE CORRELACIÓN FUNDAMENTAL (SECTORIAL)
# ==============================================================================
print("\n3. Generando Matrices de Correlación Fundamentales...")

metricas_universales = ['PER', 'Rentabilidad_Dividendo', 'Capitalizacion_Bursatil', 'Activos_Totales', 'ROA_Trimestral', 'Deuda_Total']
metricas_universales = [m for m in metricas_universales if m in df_limpio.columns and df_limpio[m].notna().sum() > 0]

# --- CORRELACIÓN GLOBAL ---
if metricas_universales:
    corr_global = df_limpio[metricas_universales].corr(method='spearman')
    
    # Exportar matriz numérica global
    corr_global.to_csv(os.path.join(Config.DIR_SALIDA, "Matriz_Correlacion_Global.csv"))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_global,
                vmin=-1.0, vmax=1.0, annot=True, fmt=".2f", cmap='RdBu', center=0, square=True,
                ax=ax, mask=np.triu(np.ones_like(corr_global, dtype=bool)))
    ax.set_title('Correlación Fundamental Global (Métricas Universales)', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.DIR_SALIDA, "03_Heatmap_Fundamental_Global.png"), dpi=300, bbox_inches='tight')
    plt.close()

# --- CORRELACIÓN SECTORIAL (BUCLE ACTUALIZADO) ---
for grupo in df_limpio['Grupo_Contable'].unique():
    if grupo == 'SIN_CLASIFICAR': continue
    
    df_sector = df_limpio[df_limpio['Grupo_Contable'] == grupo]
    
    # Exigimos un mínimo de 10 datos cruzados válidos para que la correlación tenga significancia estadística
    cols_activas = [c for c in cols_financieras if df_sector[c].notna().sum() > 10]
    if len(cols_activas) < 2: continue
    
    # Cálculo con filtrado riguroso de filas/columnas completamente vacías post-cálculo
    corr_sector = df_sector[cols_activas].corr(method='spearman').dropna(how='all', axis=0).dropna(how='all', axis=1)
    if corr_sector.empty: continue
    
    # 1. Exportar la matriz numérica (Esencial para Feature Selection)
    ruta_csv = os.path.join(Config.DIR_SALIDA, f"Matriz_Correlacion_{grupo}.csv")
    corr_sector.to_csv(ruta_csv)
    
    # 2. Renderizado del Heatmap con escalado dinámico
    tamanio = max(10, len(corr_sector.columns) * 0.6)
    fig, ax = plt.subplots(figsize=(tamanio, tamanio * 0.8))
    
    sns.heatmap(corr_sector,
                vmin=-1.0, vmax=1.0, 
                annot=True, fmt=".2f", annot_kws={"size": 8}, # Reducimos la fuente para evitar solapamientos
                cmap='RdBu', center=0, square=True,
                ax=ax, mask=np.triu(np.ones_like(corr_sector, dtype=bool)))
    
    ax.set_title(f'Correlación Fundamental (Spearman) - {grupo}', fontsize=14, fontweight='bold', pad=15)
    
    # Ajuste de etiquetas
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    
    # Guardado de la imagen
    plt.savefig(os.path.join(Config.DIR_SALIDA, f"03_Heatmap_Fundamental_{grupo}.png"), dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# 5. CROSS-CORRELATION: FUNDAMENTALES VS TÉCNICOS
# ==============================================================================
print("\n4. Ejecutando análisis Cross-Correlation (Fundamentales vs Técnicos)...")

if os.path.exists(Config.ARCHIVO_TECNICOS):
    df_tech = pd.read_csv(Config.ARCHIVO_TECNICOS)
    df_tech['Date'] = pd.to_datetime(df_tech['Date'])
    
    if 'Ticker' in df_tech.columns:
        df_tech['Ticker'] = df_tech['Ticker'].astype(str).str.strip().str.upper()
    
    print(f"   -> Dataset Técnico cargado: {df_tech.shape[0]} registros diarios.")
    
    df_limpio = df_limpio.sort_values('Date')
    df_tech = df_tech.sort_values('Date')
    
    df_cross = pd.merge_asof(
        df_limpio, df_tech, on='Date', by='Ticker', direction='backward'
    )
    
    cols_base_tech = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Precio_Apertura', 'Precio_Alto', 'Precio_Bajo', 'Precio_Cierre', 'Volumen']
    cols_tecnicas = [c for c in df_tech.columns if c not in cols_base_tech and df_tech[c].dtype in ['float64', 'int64']]
    
    if len(cols_tecnicas) > 0:
        print(f"   -> {len(cols_tecnicas)} indicadores técnicos detectados. Cruzando matrices...")
        
        if metricas_universales:
            cross_corr_global = df_cross[metricas_universales + cols_tecnicas].corr(method='spearman')
            matriz_cruce_global = cross_corr_global.loc[metricas_universales, cols_tecnicas]
            
            fig, ax = plt.subplots(figsize=(max(12, len(cols_tecnicas)*0.6), max(6, len(metricas_universales)*0.6)))
            sns.heatmap(matriz_cruce_global, vmin=-1.0, vmax=1.0, annot=True, fmt=".2f", cmap='RdBu', center=0, ax=ax)
            ax.set_title('Cross-Correlation: Fundamentales Globales vs Técnicos', fontsize=14, fontweight='bold', pad=15)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(Config.DIR_SALIDA_CROSS, "04_CrossCorr_Global.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        for grupo in df_limpio['Grupo_Contable'].unique():
            if grupo == 'SIN_CLASIFICAR': continue
            
            df_sector_cross = df_cross[df_cross['Grupo_Contable'] == grupo]
            cols_activas_fund = [c for c in cols_financieras if df_sector_cross[c].notna().sum() > 2]
            
            if len(cols_activas_fund) < 2: continue
            
            cross_corr_sector = df_sector_cross[cols_activas_fund + cols_tecnicas].corr(method='spearman')
            matriz_cruce_sector = cross_corr_sector.loc[cols_activas_fund, cols_tecnicas].dropna(how='all', axis=0).dropna(how='all', axis=1)
            
            if matriz_cruce_sector.empty: continue
            matriz_cruce_sector.to_csv(os.path.join(Config.DIR_SALIDA_CROSS, f"CrossCorr_Matriz_{grupo}.csv"))
            
            fig, ax = plt.subplots(figsize=(max(12, len(cols_tecnicas)*0.6), max(6, len(cols_activas_fund)*0.6)))
            sns.heatmap(matriz_cruce_sector, vmin=-1.0, vmax=1.0, annot=True, fmt=".2f", cmap='RdBu', center=0, ax=ax)
            ax.set_title(f'Cross-Correlation: Ratios de {grupo} vs Técnicos', fontsize=14, fontweight='bold', pad=15)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(Config.DIR_SALIDA_CROSS, f"04_CrossCorr_{grupo}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
else:
    print(f"⚠️ Aviso: No se ha encontrado el archivo técnico en {Config.ARCHIVO_TECNICOS}.")

# ==============================================================================
# 6. FIN DEL PROCESO
# ==============================================================================
t_fin = time.time()
mins = int((t_fin - tiempo_inicio) // 60)
segs = int((t_fin - tiempo_inicio) % 60)

print("\n" + "-" * 65)
print(f"✅ EDA Fundamental y Técnico completado con éxito.")
print(f"📁 Los mapas de datos faltantes se han guardado agrupados por sector en:\n   -> {Config.DIR_SALIDA_FALTANTES}/")
print(f"⏱️ Tiempo total: {mins} minutos y {segs} segundos.")
print("-" * 65)