from Analisis import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from DecisionTree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



df = data_cleaning()

# df = pd.read_csv('dataRanked.csv')

X_train, X_test, y_train, y_test = train_test_split(df.drop('blueWins',axis=1), df['blueWins'], test_size=0.2,random_state=1234)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print("X_train: ", X_train)
# print("y_train: ", y_train)

max_depth_model =20

model = DecisionTree(max_depth = max_depth_model)

model.fit(X_train, y_train)
prediction = model.predict(X_test.to_numpy())

print('Accuracy del modelo propio:', accuracy_score(prediction, y_test.to_numpy()))
print("Las 5 variables más significativas son: ",model.top5())

''' 
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
print(accuracy_score(predict, y_test))
'''

model2 = DecisionTreeClassifier(max_depth = max_depth_model)
model2.fit(X_train, y_train)
prediction = model2.predict(X_test)
print('Accuracy de modelo de sklearn:', accuracy_score(prediction, y_test.to_numpy()))


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
    
●Si experimentan overfitting, ¿qué técnica usaron para minimizarlo?

R: No experimentamos overgitting, el modelo mostro una precición aceptable entre 0.7 y 0.75.
   Sin embargo, se uso un algoritmo de podado basico, en donde se verificaba la cantidad de nodos
   asociados a un padre y el information gain de este. Si no cumplia con ciertos parametros
   definidos por nosotros, adaptados al modelo, se podaba este nodo. Esto ayuda a reducir el 
   overfitting.

●Mencione, como comentario que variables tuvieron que hacer tunning y cualquier otra consideración extra
que tuvieron que tomar en cuenta
"""