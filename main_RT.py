# Importa las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('fifa.csv',delimiter=";")

# Preprocessing
df = df.drop(['ID', 'Name', 'Photo', 'Flag', 'Club Logo','Nationality','Club','Preferred Positions'], axis=1)
df = df.dropna()
df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Feature Engineering

# 1. Age
#df['Age'] = df['Age'].astype(int)

# Definir la función lambda para convertir los valores de "Value" y "Wage" a floats
def currency_to_float(currency_string):
    if currency_string[-1] == 'M':
        return float(currency_string[1:-1]) * 1000000
    elif currency_string[-1] == 'K':
        return float(currency_string[1:-1]) * 1000
    else:
        return float(currency_string[1:])

# Aplicar la función lambda a las columnas "Value" y "Wage" del DataFrame
df['Value'] = df['Value'].apply(lambda x: currency_to_float(x))
df['Wage'] = df['Wage'].apply(lambda x: currency_to_float(x))

def cleanDF(value):
    for i in range(len(value)):
           # print(value[i])
            if len(str(value[i])) > 2 and str(value[i])[2] in ['+', '-']:
                if (value[i][2] == '+'):
                    value[i] = int(value[i][:2]) + int(value[i][3])
                elif(value[i][2] == '-'):
                    value[i] = int(value[i][:2]) - int(value[i][3])


headers = df.columns.to_list()

# Extraer el valor numérico de la columna "Agility"
for h in headers:
    cleanDF(df[h])


# Divide los datos en conjuntos de entrenamiento y prueba
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(df.drop('Potential',axis=1), df['Potential'], test_size=0.2,random_state=1234)

# Divide los datos en conjuntos de prueba y validación
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1234)


# Se valida el modelo con los datos de validación con grid search

# Define los valores de los hiperparámetros a probar
#max_depth = [2, 3, 4, 5, 6, 7, 8, 9, 10]
modelito = DecisionTreeRegressor()
param_grid = {"criterion" : ["squared_error", "friedman_mse"],
             "splitter" : ["best", "random"],
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [2, 4, 6],
                           }

grid_search = GridSearchCV(modelito, param_grid, cv=5)
grid_search.fit(X_val, y_val)

print("Resultados de Crossvalidation:")
print("Mejores parametros: ", grid_search.best_params_)
print("Mejor score: {:.2f}".format(grid_search.best_score_))

modelitoSuper = DecisionTreeRegressor(criterion= grid_search.best_params_['criterion'], 
                                      max_depth= grid_search.best_params_['max_depth'], 
                                      min_samples_split= grid_search.best_params_['min_samples_split'], 
                                      splitter= grid_search.best_params_['splitter'])


# Entrena el modelo con los datos de entrenamiento
modelitoSuper.fit(X_train, y_train)

# Realiza predicciones con los datos de prueba
y_pred = modelitoSuper.predict(X_test)

# Obtener la importancia de las características
importancias = modelitoSuper.feature_importances_

diccionario_importancias = {}
for i in range(len(importancias)):
    diccionario_importancias[modelitoSuper.feature_names_in_[i]] = importancias[i]

# Ordenar las características por importancia de mayor a menor en el diccionario
indices_ordenados = sorted(diccionario_importancias, key=diccionario_importancias.get, reverse=True)

# Seleccionar las 5 características más importantes
caracteristicas_top5 = indices_ordenados[:5]

# Imprimir las características seleccionadas
print("Las 5 características más importantes son:")
for idx in caracteristicas_top5:
    print("- Característica {}: importancia = {:.5f}".format(idx, diccionario_importancias[idx]))

# Calcula el error cuadrático medio de las predicciones
mse = mean_squared_error(y_test, y_pred)
# Imprime el error cuadrático medio
print("El error cuadrático medio es:", mse)
