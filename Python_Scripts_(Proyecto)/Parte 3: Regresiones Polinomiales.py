# Aplicación Regresión Polinomial

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def ajustar_regresion_polinomial(train_series, test_series, grado):
    # Reindexar en base a tiempo (enteros)
    X_train = np.arange(len(train_series)).reshape(-1, 1)
    y_train = train_series.values

    X_test = np.arange(len(train_series), len(train_series) + len(test_series)).reshape(-1, 1)
    y_test = test_series.values

    # Crear y ajustar modelo polinomial
    poly = PolynomialFeatures(degree=grado)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    modelo = LinearRegression()
    modelo.fit(X_poly_train, y_train)

    # Predicción
    y_pred_train = modelo.predict(X_poly_train)
    y_pred_test = modelo.predict(X_poly_test)

    # Errores
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\nGrado {grado} - RMSE Entrenamiento: {rmse_train:.2f}")
    print(f"Grado {grado} - RMSE Testeo:        {rmse_test:.2f}")

    # Visualización
    plt.figure(figsize=(14,5))
    plt.plot(train_series.index, y_train, label='Real - Entrenamiento', alpha=0.6)
    plt.plot(test_series.index, y_test, label='Real - Testeo', alpha=0.6)
    plt.plot(train_series.index, y_pred_train, label='Predicción - Entrenamiento', linestyle='--')
    plt.plot(test_series.index, y_pred_test, label='Predicción - Testeo', linestyle='--')
    plt.title(f'Regresión Polinomial (Grado {grado})')
    plt.xlabel('Fecha')
    plt.ylabel('Demanda (TotalVolume)')
    plt.legend()
    plt.tight_layout()
    plt.show()

#Evaluamos varios grados mediante un for
print('Evaluación Regresiones para Palta Convencional\n')
for grado in range(1,6):
    ajustar_regresion_polinomial(conv_train, conv_test, grado)

print('Evaluación Regresiones para Palta Orgánica\n')
for grado in range(1,6):
    ajustar_regresion_polinomial(org_train, org_test, grado)

print('\n Como conclusión respecto a las regresiones polinomiales, éstas no son capaces de representar fielmente ' \
'el comportamiento de los datos, ya que suavizan demasiado la estacionalidad en una curva continua (lo que se conoce ' \
'como "Underfitting"). Esto se refleja en su alto RMSE para todos los grados, lo que las hace imprecisas en este caso.')
