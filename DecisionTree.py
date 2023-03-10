import numpy as np
from Nodo import Nodo
from sklearn.metrics import accuracy_score
class DecisionTree(object):
    def __init__(self, min_samples = 2, max_depth = 5):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None
        self.gain_dicc = {}

    def build_tree(self, X, y, depth = 0):
        n_rows, n_cols = X.shape
        if n_rows >= self.min_samples and depth <= self.max_depth:
            best = self.best_split(X, y)
            if best['gain'] > 0:
                left = self.build_tree(X=best['df_left'][:, : -1], y = best['df_left'][:, -1], depth = depth + 1)
                right = self.build_tree(X=best['df_right'][:, : -1], y = best['df_right'][:, -1], depth = depth + 1)
                children_depth = left.depth + right.depth
                return Nodo (
                    feature = best['feature_index'],
                    threshold=best['threshold'], 
                    data_left=left, 
                    data_right=right, 
                    gain=best['gain'],
                    depth=children_depth,
                    aux_value=self.Counter(y)
                )
        return Nodo (
            value=self.Counter(y),
            depth=1
        )
    
    def Counter(self, y):
        y = y.astype('int64', casting='unsafe')
        counts = np.bincount(y)
        most_common_value = np.argmax(counts)
        return most_common_value


    def best_split(self, X, y):
        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape


        # X = X.to_numpy()
        # y = y.to_numpy()

        for index in range(n_cols):
            X_curr = X[:, index]
            # threshold = X_curr.mean()
            threshold = np.percentile(X_curr, 50)
            df = np.concatenate((X, y.reshape(1, -1).T), axis = 1)
            # Crear la partición del dataset dependiendo del threshold
            df_left = np.array([row for row in df if row[index] <= threshold])
            df_right = np.array([row for row in df if row[index] > threshold])

            if len(df_left) > 0 and len(df_right) > 0:
                # Obtener valor de la variable objetivo
                y = df[:, -1]
                y_left = df_left[:, -1]
                y_right = df_right[:, -1]

                gain = self.information_gain(y, y_right, y_left)
                if gain > best_info_gain:
                    best_split = {
                        "feature_index": index,
                        "threshold": threshold,
                        "df_left": df_left,
                        "df_right": df_right,
                        "gain": gain
                    }
                    best_info_gain = gain

        return best_split

    def information_gain(self, parent, right_child, left_child):
        size_of_data = len(parent)
        gain_right = ((len(right_child) / size_of_data) * self.gini_index(right_child))
        gain_left = ((len(left_child) / size_of_data) * self.gini_index(left_child))
        gain = self.gini_index(parent) - (gain_right + gain_left)
        return gain

    def gini_index(self, y):
        classes = np.unique(y)
        gini = 0
        for label in classes:
            probability_class = len(y[y == label]) / len(y)
            gini += probability_class ** 2
        
        return 1 - gini

    def fit(self, X, y):
        self.dataframe = X
        if type(X) is not np.ndarray:
            X = X.to_numpy()

        if type(y) is not np.ndarray:
            y = y.to_numpy()

        self.root = self.build_tree(X, y)
        self.prune(self.root)

    def predict_helper(self, x, tree):
        if tree.value != None:
            return tree.value
        feature_value = x[tree.feature]
        if feature_value <= tree.threshold:
            return self.predict_helper(x=x, tree=tree.data_left)
        
        if feature_value > tree.threshold:
            return self.predict_helper(x=x, tree=tree.data_right)
        
    def predict(self, X):
        if type(X) is not np.ndarray:
            X = X.to_numpy()

        return [self.predict_helper(x, self.root) for x in X]
    
    def prune(self, node):
        if node.value == None and node:
            self.prune(node.data_left)
            self.prune(node.data_right)
            
            right = node.data_right
            left = node.data_left

            if right and right.value == None:
                if right.depth < int(0.2 * 2 ** self.max_depth) and right.gain < 0.005:
                    node.data_right = None

            if left and left.value == None:
                if left.depth < int(0.2 *  2 ** self.max_depth) and left.gain < 0.005:
                    node.data_left = None
            
            if not(node.data_right and node.data_left):
                node.value = node.aux_value
    
    def top5(self):
        self.BFS()
        order_dicc = sorted(self.gain_dicc.items(), key=lambda x: x[1], reverse=True)[:5]
        top_5 = ""
        for x in order_dicc:
            top_5+=self.dataframe.columns[x[0]] + " "
        
        return top_5
    
    #BFS para imprimir el arbol
    def BFS(self):
        queue = []
        queue.append(self.root)
        while len(queue) > 0:
            node = queue.pop(0)
            if node.gain:
                self.gain_dicc[node.feature] = node.gain
            if node.data_left:
                queue.append(node.data_left)
            if node.data_right:
                queue.append(node.data_right)
                
                
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=True):
        return {'max_depth': self.max_depth, 'min_samples': self.min_samples}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
            
        
