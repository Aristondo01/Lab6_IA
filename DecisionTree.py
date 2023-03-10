import numpy as np
from Nodo import Nodo

class DecisionTree(object):
    def __init__(self, min_samples = 2, max_depth = 5):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.root = None

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
        # self.columns = X.columns

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
        return [self.predict_helper(x, self.root) for x in X]
    
    def prune(self, node):
        if node.value == None and node:
            right = node.data_right
            left = node.data_left
            self.prune(left)
            self.prune(right)

            if right:
                if right.depth < int(0.5 * self.max_depth) and right.gain < 0.2:
                    node.data_right = None
                    print("pode")

            if left:
                if left.depth < int(0.5 * self.max_depth) and left.gain < 0.2:
                    node.data_left = None
                    print("pode")
            
            if not(node.data_right and node.data_left):
                node.value = node.aux_value
            
        
