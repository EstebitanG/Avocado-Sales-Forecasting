#Carga de datos

import pandas as pd
import numpy as np

df4 = pd.read_csv('avocado cleaned.csv', encoding = 'UTF-8', sep=',')

print(df4)

#Preprocesamiento de datos

print(df4.info())
print(df4.isnull().sum)

#Convertir la columna Date a formato fecha (datetime)
df4['Date'] = pd.to_datetime(df4['Date'])

#Ordenar los datos por fecha
df4 = df4.sort_values('Date').reset_index(drop=True)

#Confirmar la conversión
print(df4['Date'].head())

print('El total de filas y columnas de nuestro dataframe es:\n')
print(df4.shape)

#Diágnostico valores nulos de todas las columnas
print('Diagnóstico de total de valores nulos por columna:\n')
for i in df4.columns:
    print(i+':' + str(df4[i].isnull().sum()))

#Reemplazar registros redundantes 'conventional' y 'organic'
df4['type'] = df4['type'].str.replace('conventional','Conventional')
df4['type'] = df4['type'].str.replace('organic','Organic')

#Comprobar reemplazo
print(df4['type'].unique())

#Verificar duplicados exactos en todas las columnas
duplicados_completos = df4.duplicated()
print(f'Duplicados exactos: {duplicados_completos.sum()}')

#Verificar duplicados por clave compuesta: Date + type + region
duplicados_compuesto =df4.duplicated(subset=['Date','type','region'])
print(f'Duplicados por (Date, type, region): {duplicados_compuesto.sum()}')

#Preparación de datos por región y tipo
#Filtrar datos por tipo y región
grouped_data = df4.groupby(['region','type','Date']).agg({
    'TotalVolume':'sum',
    'AveragePrice':'mean'
}).reset_index()

#Revisar los primeros datos agrupados
print(grouped_data)
