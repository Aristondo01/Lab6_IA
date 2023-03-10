from Analisis import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from DecisionTree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('fifa.csv')

# Preprocessing
df = df.drop(['ID', 'Name', 'Photo', 'Flag', 'Club Logo'], axis=1)
df = df.dropna()
df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Feature Engineering

# 1. Age
df['Age'] = df['Age'].astype(int)

# Limpiar los datos de la columna Value remplasando los datos con ints
df['Value'] = df['Value'].str.replace('€', '')
for i in range(len(df['Value'])):
    if df['Value'][i][-1] == 'M':
        df['Value'][i] = df['Value'][i][:-1]
        df['Value'][i] = float(df['Value'][i]) * 1000000
    elif df['Value'][i][-1] == 'K':
        df['Value'][i] = df['Value'][i][:-1]
        df['Value'][i] = float(df['Value'][i]) * 1000
    else:
        df['Value'][i] = df['Value'][i][:-1]
        df['Value'][i] = float(df['Value'][i])

# Limpiar los datos de la columna Wage remplasando los datos con ints
df['Wage'] = df['Wage'].str.replace('€', '')
for i in range(len(df['Wage'])):
    if df['Wage'][i][-1] == 'M':
        df['Wage'][i] = df['Wage'][i][:-1]
        df['Wage'][i] = float(df['Wage'][i]) * 1000000
    elif df['Wage'][i][-1] == 'K':
        df['Wage'][i] = df['Wage'][i][:-1]
        df['Wage'][i] = float(df['Wage'][i]) * 1000
    else:
        df['Wage'][i] = df['Wage'][i][:-1]
        df['Wage'][i] = float(df['Wage'][i])

# Importa las bibliotecas necesarias
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Carga los datos en un dataframe
#df = pd.read_csv('nombre_del_archivo.csv')

# Selecciona las características que quieres utilizar como variables predictoras
#X = df[['Overall', 'Reactions', 'Special', 'Age', 'Value', 'Wage']]

# Define la variable objetivo
#y = df['Potential']

# Divide los datos en conjuntos de entrenamiento y prueba
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(df.drop('Potential',axis=1), df['Potential'], test_size=0.2,random_state=1234)

# Divide los datos en conjuntos de prueba y validación
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1234)


# Se valida el modelo con los datos de validación con grid search

# Define los valores de los hiperparámetros a probar
#max_depth = [2, 3, 4, 5, 6, 7, 8, 9, 10]


# Crea una instancia del modelo de regresión Tree
regressor = DecisionTreeRegressor()

# Entrena el modelo con los datos de entrenamiento
regressor.fit(X_train, y_train)

# Realiza predicciones con los datos de prueba
y_pred = regressor.predict(X_test)

# Calcula el error cuadrático medio de las predicciones
mse = mean_squared_error(y_test, y_pred)

# Imprime el error cuadrático medio
print("El error cuadrático medio es:", mse)
