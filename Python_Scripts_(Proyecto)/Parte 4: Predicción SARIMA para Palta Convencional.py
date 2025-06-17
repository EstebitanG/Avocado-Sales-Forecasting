# Implementación modelos SARIMA para Palta Convencional

#Metodología Box Jenkins para determinar los parámetros del modelo SARIMA para Palta Convencional

#Test Dick Fuller para comprobar estacionariedad (ADF - Augmented Dickey Fuller) para Palta Convencional
from statsmodels.tsa.stattools import adfuller

#Aplicar test ADF sobre la demanda semanal
result_conv_series = adfuller(weekly_data_conv)

print('ADF Statistic:', result_conv_series[0])
print('p-value:', result_conv_series[1])
print('Lags usados', result_conv_series[2])
print('Número de observaciones:', result_conv_series[3])
print('Valores críticos:')
for key, value in result_conv_series[4].items():
    print(f'{key}: {value}')

print('\n La variable (cantidad demandada palta convencional) según el test ADF es estacionaria.\n')
print('Esto permite aplicar un modelo SARIMA directamente sin tener que diferenciar')

# Comprobación de estacionalidad (periodicidad) mediante gráfico de autocorrelación (ACF) para Palta Convencional
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# Asegurarse de que la serie esté bien estructurada
weekly_data_conv = weekly_data[weekly_data['type'] == 'Conventional']['TotalVolume']

# Plot de autocorrelación hasta 60 semanas
plt.figure(figsize=(12,5))
plot_acf(weekly_data_conv, lags=60)
plt.title('Autocorrelación - Palta Convencional')
plt.xlabel('Lags (semanas)')
plt.ylabel('ACF')
plt.grid()
plt.show()

print('\n Se puede observar que existen picks de estacionalidad principalmente en la semana 52, lo que indica ' \
'estacionalidad anual.')
print('Esta comprobación nos ayuda a definir el parámetro de estacionalidad "s" en nuestro modelo SARIMA')

# ACF y PACF (Autocorrelation)

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Graficar ACF y PACF
fig, axes = plt.subplots(2,1, figsize=(14,10))

# Autocorrelación ACF
plot_acf(weekly_data_conv, lags=60, ax=axes[0])
axes[0].set_title('Autocorrelación (ACF) - Palta Convencional')

# Autocorrelación parcial (PACF)
plot_pacf(weekly_data_conv, lags=60, ax=axes[1], method='ywm')
axes[1].set_title('Autocorrelación Parcial (PACF) - Palta Convencional')

plt.tight_layout()
plt.show()

print('\n Observando los gráficos de Autocorrelación, podemos definir con claridad los parámetros de nuestro modelo SARIMA:')
print('Componente regular (no estacional): ACF -> primeros 10 rezagos son significativos, pero caen rápido,' \
'lo que es típico en una MA de grado bajo, por ende hacemos MA(1) o MA(2) -> q = 1 o 2' \
'PACF -> los primeros 2 rezagos son significativos, luego caen a cero, sugiriendo un modelo AR(2) -> p= 2')
print('Componente Estacional: Ambos gráficos señalan picks en 52-53, indicando estacionalidad anual semanal. ' \
'ACF significativa en rezagos 52-53 -> Componente estacional MA -> Q = 1. ' \
'PACF significativa en rezagos 52-53 -> componente estacional AR -> P = 1')
print('SARIMA(2,0,1)(1,1,1,52) -> no hace falta diferenciar al ser la variable estacionaria' )

# Implementación SARIMA set entrenamiento y set de testeo - Palta Convencional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Ajustar modelo SARIMA
model = SARIMAX(conv_train,
                order=(2,0,1),
                seasonal_order=(1,1,1,52),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)

print(type(conv_train.index))  # Debe ser <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
print(conv_train.index.min(), conv_train.index.max())
print(conv_test.index.min(), conv_test.index.max())

assert conv_train.index.is_monotonic_increasing
assert conv_test.index.is_monotonic_increasing
assert isinstance(conv_train.index, pd.DatetimeIndex)

start_idx = len(conv_train)
end_idx = start_idx + len(conv_test) - 1

forecast = results.get_prediction(start=start_idx, end=end_idx, dynamic=False)

# Reasignar el índice correcto a las predicciones
pred_mean = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.1) # 90% intervalo de predicción
pred_mean.index = conv_test.index
conf_int.index = conv_test.index

# Graficar
plt.figure(figsize=(14,6))
plt.plot(conv_train.index, conv_train, label='Entrenamiento', color='blue')
plt.plot(conv_test.index, conv_test, label='Real (Test)', color='black')
plt.plot(pred_mean.index, pred_mean, label='Predicción', color='red')
plt.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='pink', alpha=0.3, label='Intervalo 90%')
plt.title('Predicción SARIMA - Palta Convencional (Entrenamiento)')
plt.xlabel('Fecha')
plt.ylabel('Cantidad Demandada')
plt.legend()
plt.tight_layout()
plt.show()

# Serie real y predicha en el test
y_true = conv_test
y_pred = pred_mean

# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# MAE
mae = mean_absolute_error(y_true, y_pred)

# MAPE
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Mostrar resultados
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')

# Definir fecha de inicio del zoom
start_zoom = '2020-01-01'

# Filtrar cada serie desde esa fecha
train_zoom = conv_train[conv_train.index >= start_zoom]
test_zoom = conv_test[conv_test.index >= start_zoom]
pred_zoom = pred_mean[pred_mean.index >= start_zoom]
conf_int_zoom = conf_int[conf_int.index >= start_zoom]

# Graficar con zoom desde 2020
plt.figure(figsize=(14,6))
plt.plot(train_zoom.index, train_zoom, label='Entrenamiento', color='blue')
plt.plot(test_zoom.index, test_zoom, label='Real (Test)', color='black')
plt.plot(pred_zoom.index, pred_zoom, label='Predicción', color='red')
plt.fill_between(conf_int_zoom.index,
                 conf_int_zoom.iloc[:, 0],
                 conf_int_zoom.iloc[:, 1],
                 color='pink', alpha=0.3, label='Intervalo 90%')
plt.title('Zoom Predicción SARIMA - Palta Convencional desde 2020')
plt.xlabel('Fecha')
plt.ylabel('Cantidad Demandada')
plt.legend()
plt.tight_layout()
plt.show()

print('\n El modelo SARIMA(2,0,1)(1,1,1,52) aplicado genera los siguientes resultados:')
print('\n RMSE: 15.994.651,75 -> Este estadístico mide la raíz cuadrada del promedio de los errores al cuadrado. ' \
'Su interpretación es que la predicción se desvía del valor real en torno a 16 millones de unidades ' \
'de volumen. Considerando que el volumen promedio son 100 millones de unidades, puede ser un valor aceptable.')
print('\n MAE: 13.574.314,01 -> Este estadístico es el promedio de las diferencias absolutas entre los valores ' \
'predichos y reales. Señala que en promedio el modelo se equivoca por unos 13,5 millones de unidades semanales (' \
'a nivel global).')
print('\n MAPE: 12,79% -> es el porcentaje promedio de error entre la predicción y el valor real. ' \
'Señala que en promedio las predicciones del modelo se desvían un 12,79% respecto al valor real, lo cual ' \
'puede ser razonable para series temporales comerciales con estacionalidad y ruido.')

# Código para depurar futuros errores con SARIMAX para ambos tipos de palta (Convencional y Orgánica)

# Palta Convencional

# Reiniciar índice (por si Date estaba como índice)
df4 = df4.reset_index()

# Asegurar que 'Date' sea tipo datetime
df4['Date'] = pd.to_datetime(df4['Date'])

# Agrupar por fecha y sumar los volúmenes (en caso de duplicados por región o tamaño)
weekly_data_conv = df4.groupby('Date')['TotalVolume'].sum()

# Ordenar por fecha
weekly_data_conv = weekly_data_conv.sort_index()

# Establecer frecuencia semanal explícita (rellena semanas faltantes con NaN)
weekly_data_conv = weekly_data_conv.asfreq('W')

# Eliminar semanas sin datos (por si faltan)
weekly_data_conv = weekly_data_conv.dropna()

# Verificaciones
print(type(weekly_data_conv.index))            # Esperado: DatetimeIndex
print(weekly_data_conv.index.min())            # 2015-01-04
print(weekly_data_conv.index.max())            # 2021-11-28
print(weekly_data_conv.index.freq)             # W-SUN
print(weekly_data_conv.head())

# Palta Orgánica

# Asegurar que 'Date' esté en formato datetime y depurar índice

# Paso 1: Reiniciar índice (por si Date estaba como índice)
df4 = df4.reset_index()

# Paso 2: Asegurar que 'Date' sea tipo datetime
df4['Date'] = pd.to_datetime(df4['Date'])

# Paso 3: Agrupar por fecha (suma total en cada semana, sin importar regiones ni tamaños)
weekly_data_org = df4.groupby('Date')['TotalVolume'].sum()

# Paso 4: Ordenar por fecha
weekly_data_org = weekly_data_org.sort_index()

# Paso 5: Establecer frecuencia semanal explícita
weekly_data_org = weekly_data_org.asfreq('W')

# Paso 6: Eliminar semanas sin datos (rellena huecos si los hay)
weekly_data_org = weekly_data_org.dropna()

# Paso 7: Verificaciones
print(type(weekly_data_org.index))            # Esperado: <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
print(weekly_data_org.index.min())            # Esperado: 2015-01-04 o fecha inicial real
print(weekly_data_org.index.max())            # Esperado: 2021-11-28 o fecha final real
print(weekly_data_org.index.freq)             # Esperado: W-SUN
print(weekly_data_org.head())                 # Muestra las primeras filas

# Entrenamiento del modelo SARIMAX con el 100% de datos históricos (predicción 2022)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Asegurarse que el índice sea datetime y tenga frecuencia semanal
weekly_data_conv = weekly_data_conv.copy()
weekly_data_conv.index = pd.to_datetime(weekly_data_conv.index)
weekly_data_conv = weekly_data_conv.sort_index()
weekly_data_conv = weekly_data_conv.asfreq('W') 
weekly_data_conv = weekly_data_conv.interpolate(method='linear')

# Confirmar que estamos trabajando con la serie de volumen
full_series = weekly_data_conv

# Entrenar modelo SARIMAX con todos los datos disponibles (hasta 2021)
model = SARIMAX(full_series,
                order=(2, 0, 1),
                seasonal_order=(1, 1, 1, 52),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)

# Crear fechas futuras para el año 2022 (52 semanas desde la última fecha)
last_date = full_series.index[-1]
future_index = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                             periods=52,
                             freq='W')

# Realizar predicción
forecast = results.get_forecast(steps=52)
pred_mean = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.1)  # 90% intervalo de confianza

# Reasignar el índice correcto a la predicción
pred_mean.index = future_index
conf_int.index = future_index

# Graficar
plt.figure(figsize=(14,6))
plt.plot(full_series.index, full_series, label='Datos históricos (2015–2021)', color='black')
plt.plot(pred_mean.index, pred_mean, label='Predicción 2022', color='red')
plt.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='pink', alpha=0.3, label='Intervalo 90%')
# Línea de conexión visual entre el último dato real y el primer valor predicho
plt.plot([full_series.index[-1], pred_mean.index[0]],
         [full_series.iloc[-1], pred_mean.iloc[0]],
         color='red', linestyle='--', alpha=0.6, label='Conexión real-predicción')
plt.title('Predicción 2022 - SARIMAX entrenado con datos completos')
plt.xlabel('Fecha')
plt.ylabel('Cantidad Demandada')
plt.legend()
plt.tight_layout()
plt.show()

# Graficar zoom desde 2020 en adelante
plt.figure(figsize=(14,6))

# Filtrar datos reales desde 2020
real_zoom = full_series[full_series.index >= '2020-01-01']

# Graficar datos reales desde 2020
plt.plot(real_zoom.index, real_zoom, label='Datos históricos (2020–2021)', color='black')

# Graficar predicción para 2022
plt.plot(pred_mean.index, pred_mean, label='Predicción 2022', color='red')

# Intervalo de confianza
plt.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='pink', alpha=0.3, label='Intervalo 90%')

# Línea de conexión entre 2021 y 2022
plt.plot([full_series.index[-1], pred_mean.index[0]],
         [full_series.iloc[-1], pred_mean.iloc[0]],
         color='red', linestyle='--', alpha=0.6, label='Conexión real-predicción')

# Etiquetas y leyenda
plt.title('Zoom desde 2020 - SARIMAX: Datos reales y predicción 2022')
plt.xlabel('Fecha')
plt.ylabel('Cantidad Demandada')
plt.legend()
plt.tight_layout()
plt.show()

# Crear DataFrame con la predicción de 2022 Palta Convencional
forecast_df4_palta_convencional = pd.DataFrame({
    'Fecha': pred_mean.index,
    'Prediccion_TotalVolume': pred_mean.values,
    'Lower_CI_90%': conf_int.iloc[:,0].values,
    'Upper_CI_90%': conf_int.iloc[:,1].values
})

# Establecer la columna de fecha como índice
forecast_df4_palta_convencional.set_index('Fecha', inplace=True)

# Mostrar forecasting palta convencional 2022
print(forecast_df4_palta_convencional)

# Exportación forecasting Palta Convencional a csv

#Ajustar las predicciones
forecast_df4_palta_convencional['TotalVolume'] = forecast_df4_palta_convencional['Prediccion_TotalVolume']

#Crear nuevo dataframe solo con fecha y totalvolume
export_df4 = forecast_df4_palta_convencional[['TotalVolume','Lower_CI_90%','Upper_CI_90%']].copy()
export_df4.index.name = 'Date'

#Exportar a csv
export_df4.to_csv('predicción_2022_palta_convencional.csv')

#Confirmar
print('Archivo CSV exportado exitosamente con las predicciones')
