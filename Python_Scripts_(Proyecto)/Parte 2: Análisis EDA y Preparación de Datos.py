#Análisis Exploratorio de Datos (EDA)

#Evolución de la demanda y precio a lo largo del tiempo

import matplotlib.pyplot as plt
import seaborn as sns

#Configuración de estilo
sns.set(style='whitegrid')

#Agrupar datos por fecha para analizar tendencias generales
time_series = grouped_data.groupby('Date').agg({
    'TotalVolume':'sum',
    'AveragePrice':'mean'
}).reset_index()

#Graficar evolución de la cantidad demandada
plt.figure(figsize=(14,6))
sns.lineplot(data=time_series, x='Date', y='TotalVolume', label='Demanda', color='blue')
plt.title('Evolución de la Demanda de Palta (Todas las regiones y tipos)')
plt.xlabel('Fecha')
plt.ylabel('Demanda')
plt.legend()
plt.show()

#Graficar la evolución del precio promedio
plt.figure(figsize=(14,6))
sns.lineplot(data=time_series, x='Date', y='AveragePrice', label='AveragePrice', color='green')
plt.title('Evolución del Precio Promedio de las paltas (Todas las Regiones y Tipos)')
plt.xlabel('Fecha')
plt.ylabel('Precio promedio')
plt.legend()
plt.show()

#Comparación entre tipos de paltas (convencional vs orgánica)

from matplotlib.ticker import FuncFormatter

#Agrupamos por tipo de palta
type_analysis = grouped_data.groupby(['type','Date']).agg({
    'TotalVolume':'sum',
    'AveragePrice':'mean'
}).reset_index()

#Separamos los datos por tipo de palta 

conv = type_analysis[type_analysis['type'] == 'Conventional']
org = type_analysis[type_analysis['type'] == 'Organic']

#Funcion para formatear números grandes sin abreviación
def plain_format(x, _):
    return f'{int(x):,}'.replace(',','.')

formatter = FuncFormatter(plain_format)

#Graficar en subplots
fig, axes = plt.subplots(2, 1, figsize=(14,10), sharex=True)

#Gráfico para paltas convencionales
sns.lineplot(data=conv, x='Date', y='TotalVolume', ax=axes[0], color='green')
axes[0].set_title('Demanda de Palta Convencional')
axes[0].set_ylabel('Volumen Total')
axes[0].yaxis.set_major_formatter(formatter)

#Gráfico para paltas orgánicas
sns.lineplot(data=org, x='Date', y='TotalVolume', ax=axes[1], color='orange')
axes[1].set_title('Demanda de Palta Orgánica')
axes[1].set_ylabel('Volumen Total')
axes[1].set_xlabel('Fecha')
axes[1].yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.show()

#Graficar precio promedio por tipo
plt.figure(figsize=(14,6))
sns.lineplot(data=type_analysis, x='Date', y='AveragePrice', hue='type', palette='Set2')
plt.title('Precio Promedio por Tipo de Palta (Convencional vs Orgánica)')
plt.xlabel('Fecha')
plt.ylabel('Precio Promedio')
plt.legend(title='Tipo de Palta')
plt.show()

#Relación entre precio y cantidad (gráfico de dispersión)

# Calcular correlaciones
corr_conv = conv[['TotalVolume', 'AveragePrice']].corr().iloc[0,1]
corr_org = org[['TotalVolume', 'AveragePrice']].corr().iloc[0,1]

print(f"Correlación (Palta Convencional): {corr_conv:.3f}")
print(f"Correlación (Palta Orgánica): {corr_org:.3f}")

# Graficar scatterplots con la linea de tendencia
fig, axes = plt.subplots(1, 2, figsize=(16,6), sharey=True)

# Scatterplot para palta convencional
sns.regplot(data=conv, x='TotalVolume', y='AveragePrice', ax=axes[0], scatter_kws={'alpha':0.5}, line_kws={'color':'black'})
axes[0].set_title(f'Palta Convencional: Relación Precio Cantidad (r = {corr_conv:.2f})')
axes[0].set_xlabel('Cantidad Demandada')
axes[0].set_ylabel('Precio Promedio')
axes[0].xaxis.set_major_formatter(formatter)

# Scatterplot para palta orgánica
sns.regplot(data=org, x='TotalVolume', y='AveragePrice', ax=axes[1], scatter_kws={'alpha':0.5}, line_kws={'color':'black'})
axes[1].set_title(f'Palta Orgánica: Relación Precio Cantidad (r = {corr_org:.2f})')
axes[1].set_xlabel('Cantidad Demandada')
axes[1].set_ylabel('')
axes[1].xaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.show()

print('Se puede visualizar que la demanda de palta convencional posee un coeficiente de correlación de -0.492, lo que señala que es un bien elástico, es decir, los consumidores reacciones fuertemente a cambios en el precio.\n')
print('Se puede visualizar que la demanda de palta orgánica posee un coeficiente de correlación de -0.109, lo que señala que hay una relación muy débil entre el precio de la palta orgánica y su demanda. Esto puede sugerir que la palta orgánica se mueve por otros determinantes.\n')

print('Estos análisis sugieren lo siguiente: para la palta convencional, considerar el efecto estacional y el precio como variables relevantes para predecir es adecuado.')
print('Para la palta orgánica, puede que el precio sea un predictor débil, y el modelo deba enfocarse más en patrones temporales/estacionales')

# Descomposición estacional

from statsmodels.tsa.seasonal import seasonal_decompose

# Asegurar tipo datetime y orden cronológico
df4['Date'] = pd.to_datetime(df4['Date'])
df4 = df4.sort_values('Date')

# Filtrar solo palta convencional
df4_conv = df4[df4['type'] == 'Conventional']

# Agrupar por fecha (una observación por semana por tipo) y sumar volumen
weekly_data_conv = df4_conv.groupby('Date').agg({'TotalVolume': 'sum'})

# Ahora aseguramos frecuencia semanal constante (rellena semanas faltantes con 0)
weekly_data_conv = weekly_data_conv.resample('W').sum()

# Aplicar descomposición estacional (asumimos periodo de 52 semanas)
decomp_conv = seasonal_decompose(weekly_data_conv['TotalVolume'], model='additive', period=52)

# Graficar
plt.figure(figsize=(14, 10))
decomp_conv.plot()
plt.suptitle('Descomposición Estacional - Palta Convencional', fontsize=16)
plt.tight_layout()
plt.show()

# Filtrar solo palta orgánica
df4_org = df4[df4['type'] == 'Organic']

# Agrupar por fecha (una observación por semana por tipo) y sumar volumen
weekly_data_org = df4_org.groupby('Date').agg({'TotalVolume': 'sum'})

# Ahora aseguramos frecuencia semanal constante (rellena semanas faltantes con 0)
weekly_data_org = weekly_data_org.resample('W').sum()

# Aplicar descomposición estacional (asumimos periodo de 52 semanas)
decomp_org = seasonal_decompose(weekly_data_org['TotalVolume'], model='additive', period=52)

# Graficar
plt.figure(figsize=(14, 10))
decomp_org.plot()
plt.suptitle('Descomposición Estacional - Palta Orgánica', fontsize=16)
plt.tight_layout()
plt.show()

# Preparación de los datos para modelado de series temporales

# Asegurarnos de que las fechas están ordenadas
df4 = df4.sort_values('Date')

# Agregación semanal por tipo de palta (esto para mayor granularidad, mayor disponibilidad de datos para entrenar y comportamiento estacional)
weekly_data = df4.groupby(['Date','type']).agg({
    'TotalVolume':'sum',
    'AveragePrice': 'mean'
}).reset_index()

# Separar las series temporales por tipo
df4_conv = weekly_data[weekly_data['type'] == 'Conventional'].set_index('Date')
df4_org = weekly_data[weekly_data['type'] == 'Organic'].set_index('Date')

# Copia de la variable para aplicar los modelos
weekly_data_conv = df4_conv['TotalVolume'].copy()
weekly_data_org = df4_org['TotalVolume'].copy()

# Eliminar la columna de tipo al estar implícita
df4_conv = df4_conv.drop(columns='type')
df4_org = df4_org.drop(columns='type')

# Visualización rápida (comprobación)

df4_conv['TotalVolume'].plot(figsize=(12,5), title='Demanda Semanal - Palta Convencional')
plt.ylabel('TotalVolume')
plt.show()

df4_org['TotalVolume'].plot(figsize=(12,5), title='Demanda Semanal - Palta Orgánica')
plt.ylabel('TotalVolume')
plt.show()

# Separación set entrenamiento y set testeo

# Definir fecha de corte para separación de sets
cutoff_date = '2021-01-04'

conv_train = weekly_data_conv.loc[:cutoff_date]
conv_test = weekly_data_conv.loc[cutoff_date:]

org_train = weekly_data_org.loc[:cutoff_date]
org_test = weekly_data_org.loc[cutoff_date:]

# Verificación
print(f"Convencional - Train: {conv_train.index.min().date()} a {conv_train.index.max().date()}")
print(f"Convencional - Test:  {conv_test.index.min().date()} a {conv_test.index.max().date()}")
print(f"Orgánica     - Train: {org_train.index.min().date()} a {org_train.index.max().date()}")
print(f"Orgánica     - Test:  {org_test.index.min().date()} a {org_test.index.max().date()}")

