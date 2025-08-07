# Avocado Sales Forecasting
Este proyecto forma parte de mi portfolio de Data Analytics y Data Science, donde se usa el lenguaje de programación Python aplicando técnicas estadísticas y de Machine Learning para realizar estimaciones (forecasting) de demanda de paltas en EE.UU.

# Descripción General
Este proyecto busca predecir la demanda de dos tipos de palta que se venden en EE.UU.: Palta Convencional y Palta Orgánica para el año 2022.

Los datos que se usaron fueron sacados del sitio web [Kaggle](https://www.kaggle.com/datasets/timmate/avocado-prices-2020).

Primeramente se realizó la importación del archivo csv a Python, para luego proceder con el preprocesamiento de los datos (limpieza y reordenamiento de la información), principalmente usando la librería Pandas. Posteriormente, se realiza análisis EDA para descubrir patrones iniciales en los datos, apoyándonos en las librerías Matplotlib y Seaborn principalmente. Basándose en estos resultados, se realiza una descomposición estacional para ambos tipos de palta, dando pie a la preparación de los datos para análisis de series temporales, donde fue fundamental verificar la estacionariedad de las variables a predecir usando el test Dickey-Fuller dentro de la metodología Box-Jenkins.

El análisis de series temporales se basó primeramente en la aplicación de un modelo clásico siendo SARIMA, y finalmente un modelo más reciente siendo Prophet de Facebook. Cada uno de los modelos tiene sus potencialidades y limitantes, las cuales se describen y comparan en el proyecto.

Para este proyecto se utilizó principalmente Python para cumplir con la ruta de analista y científico de datos, desde la importación de la información, su procesamiento, aplicación de modelos y exportación del forecasting. 

![Avocado_portada](https://github.com/user-attachments/assets/b530b4d5-bc87-4f6c-bd44-c9f233439b96)


