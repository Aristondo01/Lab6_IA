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

df = pd.read_csv('dataRanked.csv')

X_train, X_test, y_train, y_test = train_test_split(df.drop('blueWins',axis=1), df['blueWins'], test_size=0.2,random_state=1234)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print("X_train: ", X_train)
# print("y_train: ", y_train)

'''
model = DecisionTree()

model.fit(X_train.to_numpy(), y_train.to_numpy())
prediction = model.predict(X_test.to_numpy())

print(accuracy_score(prediction, y_test.to_numpy()))
'''
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
predict = rf.predict(X_test)
print(accuracy_score(predict, y_test))

'''

model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)
prediction = model2.predict(X_test)
print(accuracy_score(prediction, y_test.to_numpy()))
'''