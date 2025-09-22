import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class KMeansProject:
    def __init__(self, n_clusters=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.data = None
        self.labels = None

    def load_and_filter_data(self):
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)

        filtered_df = df[df[iris.feature_names[0]] > 4.5].copy()
        self.data = filtered_df
        print("🔎 Filtrlangan ma’lumotlar:", self.data.shape)

    def preprocess(self):
        self.data_scaled = self.scaler.fit_transform(self.data)

    def find_optimal_clusters(self, max_k=10):
        distortions = []
        K = range(1, max_k + 1)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(self.data_scaled)
            distortions.append(kmeans.inertia_)
        plt.figure(figsize=(6,4))
        plt.plot(K, distortions, 'bo-')
        plt.xlabel('K (klasterlar soni)')
        plt.ylabel('Distortion')
        plt.title('Elbow Method')
        plt.show()

    def build_and_fit(self):
        if self.n_clusters is None:
            raise ValueError("❗ Klasterlar sonini belgilang yoki Elbow method orqali tanlang.")
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.labels = self.model.fit_predict(self.data_scaled)
        print(f"✅ K-Means {self.n_clusters} klaster bilan o‘qitildi.")

    def visualize_clusters(self):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.data_scaled)
        plt.figure(figsize=(6,5))
        plt.scatter(reduced[:,0], reduced[:,1], c=self.labels, cmap='viridis', edgecolors='k')
        plt.title(f"K-Means ({self.n_clusters} klaster)")
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.show()

    def cluster_summary(self):
        clustered_df = self.data.copy()
        clustered_df['Cluster'] = self.labels
        summary = clustered_df.groupby('Cluster').mean()
        print("\n📊 Klaster xulosalari (o‘rtacha qiymatlar):")
        print(summary)

project = KMeansProject()
project.load_and_filter_data()      
project.preprocess()                 
project.find_optimal_clusters(8)
project.n_clusters = 3               
project.build_and_fit()              
project.visualize_clusters()         
project.cluster_summary()            
