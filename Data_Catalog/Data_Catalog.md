# Diccionario de datos para proyecto de Forecasting

Descripción General

Se detallan las columnas existentes en el dataset original (avocado cleaned.csv), las cuales fueron procesadas para el posterior entrenamiento de modelos.

### Avocado cleaned.csv

Propósito: almacena datos históricos sobre 2 tipos de palta: convencional y orgánica, junto con características de interés para su análisis y procesamiento.

| Nombre de Variable | Tipo de Dato | Descripción                                                                 |
|---------------------|--------------|------------------------------------------------------------------------------|
| Date              | object       | Fecha de la observación (formato YYYY-MM-DD). Posteriormente convertida a datetime. |
| AveragePrice      | float64      | Precio promedio por unidad de palta en la semana correspondiente.           |
| TotalVolume       | float64      | Volumen total de paltas vendidas (en unidades).                            |
| plu4046           | float64      | Volumen de palta con código PLU 4046 (pequeña).                            |
| plu4225           | float64      | Volumen de palta con código PLU 4225 (mediana).                            |
| plu4770           | float64      | Volumen de palta con código PLU 4770 (grande).                             |
| TotalBags         | float64      | Total de bolsas vendidas (suma de las siguientes tres columnas).           |
| SmallBags         | float64      | Número de bolsas pequeñas vendidas.                                        |
| LargeBags         | float64      | Número de bolsas grandes vendidas.                                         |
| XLargeBags        | float64      | Número de bolsas extragrandes vendidas.                                    |
| type              | object       | Tipo de palta: 'Conventional' o 'Organic'.                             |
| year              | int64        | Año correspondiente a la observación.                                      |
| region            | object       | Región geográfica del mercado al que corresponde la observación.           |
