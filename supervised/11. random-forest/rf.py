import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


class RandomForestRegressorCustom:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
        
    def fit(self, X, y):
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        return np.mean(predictions, axis=1)
    
class RandomForestClassifierCustom:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
        
    def fit(self, X, y):
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self
    
    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators), dtype=int)
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        return np.array([np.bincount(pred).argmax() for pred in predictions])   
    
    
if __name__ == "__main__":
    data = pd.read_csv('random-forest/dataset.csv')
    X = data.drop('quality', axis=1).values
    y = data['quality'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_regressor = RandomForestRegressorCustom(n_estimators=10, max_depth=100)
    rf_regressor.fit(X_train, y_train)
    
    y_pred = rf_regressor.predict(X_test)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    
    rf_classifier = RandomForestClassifierCustom(n_estimators=10, max_depth=100)
    rf_classifier.fit(X_train, y_train)
    
    y_pred_class = rf_classifier.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred_class))