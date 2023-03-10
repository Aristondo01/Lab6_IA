from Analisis import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from DecisionTree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from crosvalidation_DT import crosvalidation_DT



df = data_cleaning()

# df = pd.read_csv('dataRanked.csv')

X_train, X_test, y_train, y_test = train_test_split(df.drop('blueWins',axis=1), df['blueWins'], test_size=0.2,random_state=1234)

X_validate, X_test, y_validate, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1234)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print("X_train: ", X_train)
# print("y_train: ", y_train)

max_depth_model = 5

model = DecisionTree(max_depth = max_depth_model)

# model.fit(X_train, y_train)
param_grid = {'max_depth': np.arange(3, 7), 'min_samples': np.arange(2, 10)}
params = crosvalidation_DT(model, y_validate, X_validate,param_grid)

model = DecisionTree(max_depth = params['max_depth'], min_samples = params['min_samples'])

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print('\n\nAccuracy del modelo propio:', accuracy_score(prediction, y_test))
print("Las 5 variables más significativas son: ",model.top5())


print("-------------------------------------------------")


param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

model2 = DecisionTreeClassifier()
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [2, 4, 6],
}
grid_search = crosvalidation_DT(model, y_validate, X_validate, param_grid)
model2 = DecisionTreeClassifier(
    criterion = grid_search['criterion'],
    max_depth = grid_search['max_depth'],
    min_samples_split = grid_search['min_samples_split']
)

model2.fit(X_train, y_train)
prediction = model2.predict(X_test)
print('Accuracy de modelo de sklearn:', accuracy_score(prediction, y_test.to_numpy()))


print('-------------Random forest-------------')
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
print("Random forest accuracy:", accuracy_score(predict, y_test))



"""
Preguntas:

●Provea una métrica de desempeño:

R: Dado que el el dataset si esta balanceado optamos por utilizar accuracy_score

●¿Qué métrica usaron para seleccionar los features?

R: Realizamos un analisis exploratorio en busqueda de Multicolinearidad o distribuciones anomalas
Como resultado encontramos que existían muchas variaibles que daban la misma información pero desde 
la perspectiva de ambos equipos. Por ejemplo blueKills vrs redDeath la correlación es exactamente de 1,
blueAssist también compartía una correlación bastante alta con BlueKills por lo tanto tambien con redDeaths.
Como criterio buscamos todas aquellas variables que tuvieran una correlación mayor a 0.8 y menores a -0.8
y quitamos una de las variables relacionadas. De esta manera minimizariamos la cantidad de grupos 
que tendría que trabajar el árbol.

●Especifique cuales son los features que mayor importancia tomaron en laconstrucción del árbol (top 5)

R: Las 5 variables más significativas son: blueWardsDestroyed blueWardsPlaced blueGoldDiff redTotalGold redTotalJungleMinionsKilled
    Mediante un BFS se recorrio el arbol en busqueda de aquellas features que tuvieran un mayor gain
    y se guardaron en una lista. Luego se ordenaron de mayor a menor y se tomaron las 5 primeras.
    
    Es importante destacar que luego de hacer Cross Validation, se obtuvieron que los mejores parametros eran 
    max_depth: 3, min_samples: 2 por lo que el programa unicamente muestra 3 variables debido a la profundidad.
    Pero si se corriera el programa con max_depth: 5, min_samples: 2 se obtendrían las 5 variables anteriormente mencionadas.
    
●Si experimentan overfitting, ¿qué técnica usaron para minimizarlo?

R: No experimentamos overgitting, el modelo mostro una precición aceptable entre 0.7 y 0.75.
   Sin embargo, se uso un algoritmo de podado basico, en donde se verificaba la cantidad de nodos
   asociados a un padre y el information gain de este. Si no cumplia con ciertos parametros
   definidos por nosotros, adaptados al modelo, se podaba este nodo. Esto ayuda a reducir el 
   overfitting.

●Mencione, como comentario que variables tuvieron que hacer tunning y cualquier otra consideración extra
que tuvieron que tomar en cuenta

R: Las variables que se le tuvieron que hacer tunning fueron max_depth y min_samples. Se tuvo que modificar el modelo
    realizado por nosotros para poder usarlo con GridCV de forma que su interfaz fuera acorde a los modelos
    de scikit learn. 
"""