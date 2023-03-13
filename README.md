# LAB6_IA

## Decision Tree Classifier

### Provea una métrica de desempeño:

R: Dado que el el dataset si esta balanceado y el modelo es de clasificación optamos por utilizar accuracy_score

### ¿Qué métrica usaron para seleccionar los features?

R: Realizamos un analisis exploratorio en busqueda de Multicolinearidad o distribuciones anomalas
Como resultado encontramos que existían muchas variaibles que daban la misma información pero desde 
la perspectiva de ambos equipos. Por ejemplo blueKills vrs redDeath la correlación es exactamente de 1,
blueAssist también compartía una correlación bastante alta con BlueKills por lo tanto tambien con redDeaths.
Como criterio buscamos todas aquellas variables que tuvieran una correlación mayor a 0.8 y menores a -0.8
y quitamos una de las variables relacionadas. De esta manera minimizariamos la cantidad de grupos 
que tendría que trabajar el árbol.

### Especifique cuales son los features que mayor importancia tomaron en laconstrucción del árbol (top 5)

R: Las 5 variables más significativas son: blueWardsDestroyed blueWardsPlaced blueGoldDiff  redTotalGold redTotalJungleMinionsKilled
Mediante un BFS se recorrio el arbol en busqueda de aquellas features que tuvieran un mayor gain
y se guardaron en una lista. Luego se ordenaron de mayor a menor y se tomaron las 5 primeras.

Es importante destacar que luego de hacer Cross Validation, se obtuvieron que los mejores parametros eran 
max_depth: 3, min_samples: 2 por lo que el programa unicamente muestra 3 variables debido a la profundidad.
Pero si se corriera el programa con max_depth: 5, min_samples: 2 se obtendrían las 5 variables anteriormente mencionadas.
    
### Si experimentan overfitting, ¿qué técnica usaron para minimizarlo?

R: No experimentamos overfitting, el modelo mostro una precición aceptable entre 0.7 y 0.75.
   Sin embargo, se uso un algoritmo de podado basico, en donde se verificaba la cantidad de nodos
   asociados a un padre y el information gain de este. Si no cumplia con ciertos parametros
   definidos por nosotros, adaptados al modelo, se podaba este nodo. Esto ayuda a reducir el 
   overfitting.

### Mencione, como comentario que variables tuvieron que hacer tunning y cualquier otra consideración extra que tuvieron que tomar en cuenta

R: Las variables que se le tuvieron que hacer tunning fueron max_depth y min_samples. Se tuvo que modificar el modelo realizado por nosotros para poder usarlo con GridCV de forma que su interfaz fuera acorde a los modelos
de scikit learn. 

### ¿Cuál implementación fue mejor? ¿Por qué?
Nuestra implementación, inicialmente, fue ligeramente mejor que la de scikitlearn, al observar que nuestro modelo tenía un accuracy de 0.715 y el de ellos 0.711. Esta diferencia se pudo deber a que nosotros hicimos un modelo
dedicado al dataset empleado, de forma que los thresholds se ajustaran a la data que teníamos. Además, la
librería usa un random_state que afectas las particiones realizadas, cosa que no hacemos nosotros.
Sin embargo, luego de hacer cross validation con el modelo de la librería, este nos superó, siendo su accuracy
de 0.72. Muy probablemente al hacer el cross, se logró mitigar la parte aleatoria, haciendo que se tuvieran
mejores particiones.


## Decision Tree Regression



### Provea una métrica de desempeño:

R: Dado que es una regresion el error cuadratico medio es la metrica de desempeño

### ¿Qué métrica usaron para seleccionar los features?

R: Con respecto a los datos se busco solo realizar el encoding de variables y el escalado de las mismas.
    Como por ejemplo el Value y Wage que se convirtieron a float y se escalaron ya que eran estrings con $ y K o M para denotar
    el monto que se le estaba dando al jugador.Adicional a cada variable se verifico que fiera un numeor ya que muchas venian con
    un poperador +/-. Por ejemplo: 41+2, 71-2, 21+3, etc. Se decidio tomar el primer numero y sumarle o restarle el segundo.

### Especifique cuales son los features que mayor importancia tomaron en laconstrucción del árbol (top 5)

R: Las 5 variables más significativas son: Value, Age, Overall, CF ,SlidibngTackle
    
    
### Si experimentan overfitting, ¿qué técnica usaron para minimizarlo?

R: No experimentamos overfitting, el modelo mostro una precición aceptable de 0.87 esto debido 
    que se hizo un buen tunning de los hiperparametros y se uso cross validation para validar el modelo.

### Mencione, como comentario que variables tuvieron que hacer tunning y cualquier otra consideración extra
que tuvieron que tomar en cuenta

R: Las variables que se le tuvieron que hacer tunning fueron max_depth, min_samples_split, splitter y criterion.
    Se decidio hacer tunning de estas variables ya que son las que mas afectan el modelo y se puede ver en la documentacion
    de sklearn que estas variables son las que mas afectan el modelo.
