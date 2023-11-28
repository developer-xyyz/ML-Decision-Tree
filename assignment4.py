#Ahnaf Ahmad
#1001835014

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import time
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None

class DecisionTree:

    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def _finished(self, depth):
        if depth >= self.max_depth:
            return True
        elif self.class_labels == 1:
            return True
        elif self.n_samples < self.min_samples_split:
            return True
        return False
    
    def _entropy(self, y):
        proportions = np.bincount(np.array(y, dtype='int64')) / len(y)
        entropy = 0
        for p in proportions:
            if p > 0:
                entropy = entropy - (p * np.log2(p))
                return entropy

    def _split(self, X, thresh):
        left_idx = np.where(X <= thresh)[0]
        right_idx = np.where(X > thresh)[0]
        return left_idx, right_idx

    def _info_gain(self, X, y, thresh):
        size = len(y)
        parent = self._entropy(y)
        left_idx, right_idx = self._split(X, thresh)
        size_left, size_right = len(left_idx), len(right_idx)

        if (size_left == 0) or (size_right == 0): 
            return 0
        
        child = ((size_left / size) * self._entropy(y[left_idx])) + ((size_right / size) * self._entropy(y[right_idx]))
        info_gain = parent - child
        return info_gain

    def _best_split(self, X, y, features):
        best_feature, best_thresh = None, None
        best_score = -1

        for feature in features:
            X_feat = X[:, feature]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._info_gain(X_feat, y, thresh)
                if score > best_score:
                    best_score, best_feature, best_thresh = score, feature, thresh

        return best_feature, best_thresh

    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.class_labels = len(np.unique(y))

        if self._finished(depth):
            label_counts = Counter(y)
            most_common_label = label_counts.most_common(1)[0][0]
            return Node(value=most_common_label)

        rando_features = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, rando_features)
        left_idx = np.where(X[:, best_feature] <= best_thresh)[0]
        right_idx = np.where(X[:, best_feature] > best_thresh)[0]
        left_childNode = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_childNode = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feature, best_thresh, left_childNode, right_childNode)

    def _traverse_tree(self, X, node):
        while not node.is_leaf():
            if X[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        predictions = []
        for x in X:
            prediction = self._traverse_tree(x, self.root)
            predictions.append(prediction)
        return np.array(predictions)
    
    def print_tree(self, node=None, space=''):
        if node is None:
            node = self.root
        if node.is_leaf():
            print(space + f'[ Winner: {node.value} ]')
            return
        #recursivily print each child node
        print(space + f'[ Feature: {node.feature}, Threshold: {node.threshold} ]')
        print(space + 'Left:')
        self.print_tree(node.left, space + '     ')

        print(space + 'Right:')
        self.print_tree(node.right, space + '     ')
    

start_time = time.time()

#train the tree using train set
df = pd.read_csv('btrain.csv')
df.replace('?', 0, inplace=True)
df = df.apply(pd.to_numeric)
X_train = df.iloc[:, :-1].values
y_train = df.iloc[:, -1].values

dt = DecisionTree(max_depth=2)

dt.fit(X_train, y_train)
print('Tree Visualization:')
dt.print_tree()
print('')

#test the accuracy of the tree using validate set
df2 = pd.read_csv('bvalidate.csv')
df2.replace('?', 0, inplace=True)
df2 = df2.apply(pd.to_numeric)
X_test2 = df2.iloc[:, :-1].values
y_test = df2.iloc[:, -1].values

y_pred = dt.predict(X_test2)
acc = accuracy_score(y_test, y_pred)
print(f'Training Accuracy: {acc*100}%')

#predict and fill the ? of the test set
df3 = pd.read_csv('btest.csv')
df3.replace('?', 0, inplace=True)
df3 = df3.apply(pd.to_numeric)
X_test = df3.iloc[:, :-1].values
y_true = df3.iloc[:, -1].values

y_pred = dt.predict(X_test)

data = np.concatenate((X_test, y_pred.reshape(-1, 1)), axis=1)
columns = list(df3.columns[:-1]) + ['winner']
df4 = pd.DataFrame(data=data, columns=columns)

df4.to_csv('btest_v2.csv', index=False)

print(f"Execution time: {round((time.time() - start_time),2)} seconds.")