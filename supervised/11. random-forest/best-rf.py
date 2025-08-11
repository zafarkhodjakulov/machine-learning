import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


class RandomForestRegressorCustom(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
        
    def fit(self, X, y):
        self.trees = []
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
    

class RandomForestClassifierCustom(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
        
    def fit(self, X, y):
        self.trees = []
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
    
    param_grid = {
        'n_estimators': [5, 10, 20, 50],
        'max_depth': [None, 5, 10, 20, 50, 100]
    }
    
    grid_reg = GridSearchCV(RandomForestRegressorCustom(), param_grid, scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=3)
    grid_reg.fit(X_train, y_train)
    best_reg = grid_reg.best_estimator_
    y_pred = best_reg.predict(X_test)
    print("Best params (Regressor):", grid_reg.best_params_)
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    
    grid_clf = GridSearchCV(RandomForestClassifierCustom(), param_grid, scoring='accuracy', cv=3)
    grid_clf.fit(X_train, y_train)
    best_clf = grid_clf.best_estimator_
    y_pred_class = best_clf.predict(X_test)
    print("Best params (Classifier):", grid_clf.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred_class))
