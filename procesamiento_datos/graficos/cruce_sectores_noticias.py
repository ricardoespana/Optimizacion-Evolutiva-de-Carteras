import pandas as pd

# 1. Cargar
df_noticias = pd.read_csv('dataset_2021_2026_eleconomista.csv')
df_universo = pd.read_csv('df_u_lseg.csv')

# 2. LIMPIEZA CRÍTICA: Estandarizar Tickers
# Quitamos ".MC" si existe en ambos para que sean comparables
df_noticias['Ticker_Clean'] = df_noticias['Ticker'].astype(str).str.replace('.MC', '', regex=False).str.strip().str.upper()
df_universo['Ticker_Clean'] = df_universo['RIC'].astype(str).str.replace('.MC', '', regex=False).str.strip().str.upper()

# 3. Diagnóstico (Ver qué está pasando)
print(f"Tickers únicos en noticias: {df_noticias['Ticker_Clean'].nunique()}")
print(f"Tickers únicos en universo: {df_universo['Ticker_Clean'].nunique()}")

# 4. Cruce forzado
# Primero eliminamos cualquier columna 'Sector' vieja para evitar conflictos
if 'Sector' in df_noticias.columns:
    df_noticias = df_noticias.drop(columns=['Sector'])

# Merge usando la columna limpia
df_final = df_noticias.merge(
    df_universo[['Ticker_Clean', 'Sector']], 
    on='Ticker_Clean', 
    how='left'
)

# 5. Auditoría post-merge
print("-" * 30)
nulos = df_final['Sector'].isna().sum()
print(f"Valores nulos en Sector tras el cruce: {nulos}")

if nulos > 0:
    print("Muestreo de Tickers sin sector (ejemplo):")
    print(df_final[df_final['Sector'].isna()]['Ticker'].unique()[:10])

# 6. Guardar
df_final.to_csv('dataset_2021_2026_eleconomista.csv', index=False, encoding='utf-8-sig')