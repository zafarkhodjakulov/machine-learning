import numpy as np
import pandas as pd
from scipy.stats import mode 
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('knn/telecom.csv')

df.drop(columns=['customerID'])

categorical = df.select_dtypes(include=['object']).columns
numerical = df.select_dtypes(exclude=['object']).columns

label = LabelEncoder()
for col in categorical:
    if df[col].dtype == 'object':
        df[col] = label.fit_transform(df[col])
        
x = df.drop(columns=['Churn']).values
y = df['Churn'].values
        
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)


class KNN:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, x_train, y_train):
        self.x = x_train
        self.y = y_train

    def predict(self, x_test):
        dists = np.sqrt(((x_test[:, np.newaxis, :] - self.x[np.newaxis, :, :]) ** 2).sum(axis=2))
        neighbors_idx = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        neighbor_labels = self.y[neighbors_idx]
        predictions, _ = mode(neighbor_labels, axis=1) # Classification
        # predictions = np.mean(neighbor_labels, axis=1) # Regression
        return predictions.ravel()  
    
    
model = KNN(n_neighbors=100)
model.fit(X_train, y_train)
pred = model.predict(X_test)
df1 = pd.DataFrame(data=pred, columns=['predictions'])
print(pred)



# class KNN:
#     def __init__(self, n_neighbors):
#         self.n_neighbors = n_neighbors

#     def fit(self, X, y):
#         self.X = X
#         self.y = y
#         return self

#     def predict(self, X):
#         predictions = []
#         for i in X:
#         # shape of (n, m)
#         # n - observations
#         # m - number of features
#             distances = np.sqrt(np.sum((i - self.X) ** 2, axis=1))
#             idx = np.argsort(distances)[:self.n_neighbors]
#             classes = self.y[idx]
#             cntr = Counter(classes)
#             predictions.append(cntr.most_common(1)[0][0].item())
#         return distances