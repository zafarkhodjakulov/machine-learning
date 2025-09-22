import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class EnsembleProject:
    def __init__(self, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    def load_and_filter_data(self):
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target

        filtered_df = df[df['target'].isin([0,1])].copy()
        X = filtered_df.drop('target', axis=1)
        y = filtered_df['target']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    def preprocess(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def build_models(self):
        self.models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=self.random_state),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=self.random_state),
            "Voting Ensemble": VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(max_iter=200)),
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
                    ('svc', SVC(probability=True))
                ],
                voting='soft'
            )
        }

    def train_and_evaluate(self):
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            self.results[name] = acc
            print(f"🔎 {name} natijasi:")
            print(classification_report(self.y_test, y_pred))

    def summary(self):
        results_df = pd.DataFrame(list(self.results.items()), columns=["Model", "Accuracy"])
        print("\n📊 Natijalar xulosasi:")
        print(results_df.sort_values(by="Accuracy", ascending=False))


project = EnsembleProject()
project.load_and_filter_data()
project.preprocess()
project.build_models()
project.train_and_evaluate()
project.summary()
