import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("🔎 Datasetning dastlabki 5 qatori:")
df.head()

filtered_df = df[df['target'].isin([0,1])].copy()


X = filtered_df.drop('target', axis=1)
y = filtered_df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

y_pred = svm_model.predict(X_test_scaled)

print("\n📊 Klassifikatsiya hisoboti:")
print(classification_report(y_test, y_pred))

print("\n📉 Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(6,5))
plt.scatter(X_test_scaled[:,0], X_test_scaled[:,1], c=y_pred, cmap='coolwarm', edgecolors='k')
plt.title("SVM Klassifikatsiya Natijalari")
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.show()
