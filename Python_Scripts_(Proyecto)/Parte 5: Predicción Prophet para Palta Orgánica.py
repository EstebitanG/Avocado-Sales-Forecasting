#Metodología Box Jenkins para determinar los parámetros del modelo SARIMA para Palta Orgánica

#Test Dick Fuller para comprobar estacionariedad (ADF - Augmented Dickey Fuller) para Palta Orgánica
from statsmodels.tsa.stattools import adfuller

#Aplicar test ADF sobre la demanda semanal
result_org_series = adfuller(weekly_data_org)

print('ADF Statistic:', result_org_series[0])
print('p-value:', result_org_series[1])
print('Lags usados', result_org_series[2])
print('Número de observaciones:', result_org_series[3])
print('Valores críticos:')
for key, value in result_org_series[4].items():
    print(f'{key}: {value}')

print('\n La variable al ser no estacionaria, aplicamos diferenciación\n')

# Serie diferenciada para palta orgánica

# Asegurarse de que el índice sea datetime
weekly_data_org = weekly_data_org.copy()
weekly_data_org.index = pd.to_datetime(weekly_data_org.index)
weekly_data_org = weekly_data_org.sort_index()

# Diferenciación simple 
org_diff = weekly_data_org.diff().dropna()

#Verificación 
print(type(org_diff.index))

#Visualizar la serie diferenciada
plt.figure(figsize=(14,5))
plt.plot(org_diff, label='Demanda diferenciada Orgánica')
plt.title('Serie diferenciada - Palta Orgánica')
plt.xlabel('Fecha')
plt.ylabel('Diferencia de TotalVolume')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#Comprobar estacionariedad nuevamente con variable diferenciada
result_diff = adfuller(org_diff)

print('ADF Statistic (diferenciada):', result_diff[0])
print('p-value:', result_diff[1])
print('Lags usados', result_diff[2])
print('Valores críticos:')
for key, value in result_diff[4].items():
    print(f'{key}: {value}')

print('\n Nota: el aumento del "ruido" en la serie diferenciada (esos picks exagerados en el gráfico) no corresponden ' \
'a ruido aleatorio, mas bien son variaciones reales de la demanda entre semanas, lo que es esperable en series con ' \
'estacionalidad. Esto no afecta a la implementación del modelo SARIMA.')
print('\n SARIMA modela la estructura de la serie original, no la serie diferenciada. Esto lo hace ajustando los ' \
'parámetros para revertir esa transformación internamente (integra la serie)')

# Comprobación de estacionalidad (periodicidad) mediante gráfico de autocorrelación (ACF) para Palta Orgánica
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# Asegurarse de que la serie esté bien estructurada
weekly_data_org = weekly_data[weekly_data['type'] == 'Organic']['TotalVolume']

# Palta Orgánica (no diferenciada)
plt.figure(figsize=(12,5))
plot_acf(weekly_data_conv, lags=60)
plt.title('Autocorrelación - Palta Orgánica')
plt.xlabel('Lags (semanas)')
plt.ylabel('ACF')
plt.grid()
plt.show()

print('\n Se puede observar que existen picks de estacionalidad principalmente en la semana 52, lo que indica ' \
'estacionalidad anual.')
print('Esta comprobación nos ayuda a definir el parámetro de estacionalidad "s" en nuestro modelo SARIMA')

# Comprobación de estacionalidad (periodicidad) mediante gráfico de autocorrelación (ACF) para Palta Orgánica 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Establecer número de lags a mostrar
lags = 52

# Gráfico ACF (para determinar q)
plt.figure(figsize=(14,4))
plot_acf(org_diff, lags=lags, alpha=0.05)
plt.title('ACF - Palta Orgánica (Serie Diferenciada)')
plt.grid(True)
plt.show()

# Gráfico PACF (para determinar p)
plt.figure(figsize=(14,4))
plot_pacf(org_diff, lags=lags, alpha=0.05, method='ywm')
plt.title('PACF - Palta Orgánica (Serie Diferenciada)')
plt.grid(True)
plt.show()

print('\n Observando los gráficos de Autocorrelación, podemos definir con claridad los parámetros de nuestro modelo SARIMA:')
print('Componente regular (no estacional): ACF -> primeros 3 rezagos son significativos, pero caen rápido,' \
'lo que es típico en una MA de grado bajo, por ende hacemos MA(2) -> q = 2' \
'PACF -> los primeros 3 rezagos son significativos, luego caen a cero, sugiriendo un modelo AR(2) -> p= 2')
print('Componente Estacional: Ambos gráficos señalan picks en 52-53, indicando estacionalidad anual semanal. ' \
'ACF significativa en rezagos 52-53 -> Componente estacional MA -> Q = 2. ' \
'PACF significativa en rezagos 52-53 -> componente estacional AR -> P = 2')
print('SARIMA(2,1,2)(2,1,2,52) -> siendo el 1 la diferenciación de la variable.' )

# Implementación SARIMA set entrenamiento y set de testeo - Palta Orgánica

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Ajustar modelo SARIMA para palta orgánica
model = SARIMAX(org_train,
                order=(2,1,2),
                seasonal_order=(2,1,2,52),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)

# Verificaciones de índice
print(type(org_train.index))  # Esperado: <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
print(org_train.index.min(), org_train.index.max())
print(org_test.index.min(), org_test.index.max())

assert org_train.index.is_monotonic_increasing
assert org_test.index.is_monotonic_increasing
assert isinstance(org_train.index, pd.DatetimeIndex)

# Definir inicio y fin de predicción
start_idx = len(org_train)
end_idx = start_idx + len(org_test) - 1

# Generar predicciones para el set de testeo
forecast = results.get_prediction(start=start_idx, end=end_idx, dynamic=False)

# Reasignar el índice correcto a las predicciones
pred_mean = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.1)  # 90% intervalo de confianza
pred_mean.index = org_test.index
conf_int.index = org_test.index

# Gráfico general
plt.figure(figsize=(14,6))
plt.plot(org_train.index, org_train, label='Entrenamiento', color='blue')
plt.plot(org_test.index, org_test, label='Real (Test)', color='black')
plt.plot(pred_mean.index, pred_mean, label='Predicción', color='red')
plt.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='pink', alpha=0.3, label='Intervalo 90%')
plt.title('Predicción SARIMA - Palta Orgánica (Entrenamiento)')
plt.xlabel('Fecha')
plt.ylabel('Cantidad Demandada')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluación de desempeño
y_true = org_test
y_pred = pred_mean

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'MAPE: {mape:.2f}%')

# Zoom desde 2020
start_zoom = '2020-01-01'
train_zoom = org_train[org_train.index >= start_zoom]
test_zoom = org_test[org_test.index >= start_zoom]
pred_zoom = pred_mean[pred_mean.index >= start_zoom]
conf_int_zoom = conf_int[conf_int.index >= start_zoom]

plt.figure(figsize=(14,6))
plt.plot(train_zoom.index, train_zoom, label='Entrenamiento', color='blue')
plt.plot(test_zoom.index, test_zoom, label='Real (Test)', color='black')
plt.plot(pred_zoom.index, pred_zoom, label='Predicción', color='red')
plt.fill_between(conf_int_zoom.index,
                 conf_int_zoom.iloc[:, 0],
                 conf_int_zoom.iloc[:, 1],
                 color='pink', alpha=0.3, label='Intervalo 90%')
plt.title('Zoom Predicción SARIMA - Palta Orgánica desde 2020')
plt.xlabel('Fecha')
plt.ylabel('Cantidad Demandada')
plt.legend()
plt.tight_layout()
plt.show()

print('\n Reflexión sobre esta implementación: al usar los parámetros definidos por ACF-PACF (Box Jenkins),' \
'tenemos buenos resultados, pero no óptimos. Si bien los indicadores de desviación son aceptables, estos se ' \
'pueden minimizar aún más. Además, lo que genera más preocupación es la suavización excesiva de este SARIMA respecto ' \
'a los datos reales de 2021. No logra capturar el comportamiento "brusco" de los datos originales. Es por ello, que' \
'se intentará aplicar el modelo Prophet para este caso.')

# Implementación Prophet para Palta Orgánica set de entrenamiento y testeo

# Preparar dataframe para prophet
prophet_train = org_train.reset_index()
prophet_train.columns = ['ds', 'y']

from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Hacemos log-transform
prophet_train_log = prophet_train.copy()
prophet_train_log['y'] = np.log1p(prophet_train_log['y'])  # log(1 + y)

# Inicializar modelo 
model_prophet = Prophet(weekly_seasonality=True, yearly_seasonality=True)

# Entrenar 
model_prophet.fit(prophet_train_log, algorithm='LBFGS')

# Crear DataFrame futuro (número de semanas en test)
future = model_prophet.make_future_dataframe(periods=len(org_test), freq='W')

# Predicción
forecast = model_prophet.predict(future)

# ------------------ GRÁFICO 1: PREDICCIÓN SOLO EN PERIODO DE TEST ------------------ #

# Convertimos a escala real las columnas de interés
forecast['yhat_real'] = np.expm1(forecast['yhat'])
forecast['yhat_lower_real'] = np.expm1(forecast['yhat_lower'])
forecast['yhat_upper_real'] = np.expm1(forecast['yhat_upper'])

# Filtrar predicción solo del periodo de test (2021)
forecast_test = forecast.set_index('ds').loc[org_test.index]

# Concatenar datos reales (entrenamiento + test)
real_full = pd.concat([prophet_train.set_index('ds')['y'], org_test])
real_full = real_full.sort_index()

plt.figure(figsize=(14, 6))

# Línea continua con datos reales
plt.plot(real_full.index, real_full.values, label='Datos reales', color='black')

# Línea de predicción Prophet en 2021
plt.plot(forecast_test.index, forecast_test['yhat_real'], label='Predicción Prophet (2021)', color='green')

# Banda de incertidumbre (intervalo de predicción)
plt.fill_between(
    forecast_test.index,
    forecast_test['yhat_lower_real'],
    forecast_test['yhat_upper_real'],
    color='green',
    alpha=0.2,
    label='Intervalo de predicción'
)

plt.title('Predicción con Prophet - Palta Orgánica (Entrenamiento)')
plt.xlabel('Fecha')
plt.ylabel('Cantidad Demandada')
plt.legend()
plt.tight_layout()
plt.show()
# ------------------ MÉTRICAS Y PREPARACIÓN PARA ZOOM ------------------ #

# Extraer solo valores de test
forecast_filtered = forecast.set_index('ds').loc[org_test.index]
y_pred_prophet_real = np.expm1(forecast_filtered['yhat'])

# Métricas reales
rmse_prophet = np.sqrt(mean_squared_error(org_test, y_pred_prophet_real))
mae_prophet = mean_absolute_error(org_test, y_pred_prophet_real)
mape_prophet = np.mean(np.abs((org_test - y_pred_prophet_real) / org_test)) * 100

print(f'RMSE Prophet: {rmse_prophet:.2f}')
print(f'MAE Prophet: {mae_prophet:.2f}')
print(f'MAPE Prophet: {mape_prophet:.2f}')

# ------------------ GRÁFICO 2: ZOOM CON INTERVALOS ------------------ #

# Intervalos en escala real
yhat = np.expm1(forecast_filtered['yhat'])
yhat_lower = np.expm1(forecast_filtered['yhat_lower'])
yhat_upper = np.expm1(forecast_filtered['yhat_upper'])

start_zoom = '2020-01-01'

plt.figure(figsize=(14, 6))
plt.plot(org_train[org_train.index >= start_zoom], label='Entrenamiento', color='blue')
plt.plot(org_test[org_test.index >= start_zoom], label='Real (Test)', color='black')
plt.plot(yhat[yhat.index >= start_zoom], label='Predicción Prophet', color='green')
plt.fill_between(yhat[yhat.index >= start_zoom].index,
                 yhat_lower[yhat.index >= start_zoom],
                 yhat_upper[yhat.index >= start_zoom],
                 color='green', alpha=0.2, label='Intervalo de predicción')
plt.title('Zoom Predicción Prophet - Palta Orgánica desde 2020')
plt.xlabel('Fecha')
plt.ylabel('Cantidad Demandada')
plt.legend()
plt.tight_layout()
plt.show()

print('\n Al implementar Prophet como modelo de forecasting para la palta orgánica, vale la pena comparar con ' \
'nuestro modelo SARIMA antes implementado usando los indicadores de dispersión: ')
print('Respecto al RMSE, Prophet arroja 616.557 vs SARIMA que arroja 510.805 -> esto señala que prophet comete errores ' \
'absolutos más altos en predicción, es decir, estima valores más extremos. Sin embargo, no son para nada desproporcionados ' \
'dado el contexto del dataset.')
print('Respecto al MAE, Prophet arroja 510.154 vs SARIMA que arroja 410.863 -> esto señala que SARIMA tiene menor ' \
'error absoluto medio. Tiene la misma interpretación al RMSE.')
print('Respecto al MAPE, Prophet arroja 10,75% vs SARIMA que arroja 8,26% -> esto señala que SARIMA es más preciso ' \
'en términos relativos, es decir, al momento de generalizar la predicción.')

print('\n En términos generales, podemos decir lo siguiente: SARIMA le gana en todos los indicadores de dispersión ' \
'a Prophet, pero Prophet modela de mejor manera el comportamiento estacional brusco de la palta orgánica. ' \
'La pregunta es: ¿cuál elegir? -> SARIMA es mejor si importa la precisión numérica promedio y si queremos ' \
'evitar sobreestimaciones o subestimaciones sistemáticas en las ventas, es decir, apuntar a tener un stock de ' \
'inventario estable. Elegimos Prophet si necesitamos un modelo reactivo, capaz de ajustarse a estacionalidades ' \
'cambiantes que detecta comportamientos inesperados o picks de demanda.')

# Implementación Prophet para predicción demanda Palta Orgánica 2022

from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Preparación de datos ---
# Re-definición de variables para evitar ambiguedades en el dataframe
df4_org = weekly_data[weekly_data['type'] == 'Organic'].copy()
df4_org = df4_org.set_index('Date')
weekly_data_org = df4_org[['TotalVolume']].rename(columns={'TotalVolume': 'y'})

weekly_data_org.index = pd.to_datetime(weekly_data_org.index)
weekly_data_org = weekly_data_org.sort_index()
weekly_data_org = weekly_data_org.asfreq('W') 
weekly_data_org = weekly_data_org.interpolate(method='linear')

prophet_full = weekly_data_org.reset_index()
prophet_full.columns = ['ds', 'y']
prophet_full_log = prophet_full.copy()
prophet_full_log['y'] = np.log1p(prophet_full_log['y'])

# --- Entrenamiento modelo ---
model_full = Prophet(weekly_seasonality=True, yearly_seasonality=True)
model_full.fit(prophet_full_log, algorithm='LBFGS')

# --- Predicción 2022 ---
future_full = model_full.make_future_dataframe(periods=52, freq='W')
forecast_full = model_full.predict(future_full)

forecast_full['yhat_real'] = np.expm1(forecast_full['yhat'])
forecast_full['yhat_lower_real'] = np.expm1(forecast_full['yhat_lower'])
forecast_full['yhat_upper_real'] = np.expm1(forecast_full['yhat_upper'])

# --- Gráfico completo ---
plt.figure(figsize=(14,6))
plt.plot(weekly_data_org, label='Histórico Real', color='black')

# Predicción desde 2022
forecast_2022 = forecast_full[forecast_full['ds'] >= '2022-01-01']
plt.plot(forecast_2022['ds'], forecast_2022['yhat_real'], label='Predicción Prophet 2022', color='green')
plt.fill_between(forecast_2022['ds'], 
                 forecast_2022['yhat_lower_real'], 
                 forecast_2022['yhat_upper_real'], 
                 color='green', alpha=0.2, label='Intervalo de predicción')

# --- Línea discontinua de conexión entre 2021 y 2022 ---
last_2021_date = weekly_data_org.index.max()
last_2021_value = weekly_data_org.loc[last_2021_date, 'y']
first_2022_date = forecast_2022['ds'].min()
first_2022_value = forecast_2022[forecast_2022['ds'] == first_2022_date]['yhat_real'].values[0]

plt.plot([last_2021_date, first_2022_date],
         [last_2021_value, first_2022_value],
         color='green', linestyle='dashed', linewidth=2)

# --- Estética ---
plt.title('Predicción 2022 - Palta Orgánica con Prophet')
plt.xlabel('Fecha')
plt.ylabel('Cantidad Demandada')
plt.legend()
plt.tight_layout()
plt.show()

# Crear DataFrame con predicción de 2022 Palta Orgánica
forecast_df4_palta_organica = pd.DataFrame({
    'Fecha': forecast_2022['ds'],
    'Prediccion_TotalVolume': forecast_2022['yhat_real'].values,
    'Lower_CI_90%': forecast_2022['yhat_lower_real'].values,
    'Upper_CI_90%': forecast_2022['yhat_upper_real'].values
})

# Establecer la columna de fecha como índice
forecast_df4_palta_organica.set_index('Fecha', inplace=True)

# Mostrar forecasting palta orgánica 2022
print(forecast_df4_palta_organica)

# Exportación forecasting Palta Orgánica a csv

# Ajustar nombre de columna final para compatibilidad
forecast_df4_palta_organica['TotalVolume'] = forecast_df4_palta_organica['Prediccion_TotalVolume']

# Crear nuevo dataframe solo con columnas necesarias
export_df4_org = forecast_df4_palta_organica[['TotalVolume', 'Lower_CI_90%', 'Upper_CI_90%']].copy()
export_df4_org.index.name = 'Date'

# Exportar a CSV
export_df4_org.to_csv('predicción_2022_palta_organica.csv')

# Confirmar exportación
print('Archivo CSV exportado exitosamente con las predicciones de palta orgánica 2022')
