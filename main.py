from Analisis import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score




df = data_cleaning()

X_train, X_test, y_train, y_test = train_test_split(df.drop('blueWins',axis=1), df['blueWins'], test_size=0.2,random_state=1234)

print("X_train: ", X_train)
print("y_train: ", y_train)
