# 1️⃣ Kutubxonalarni chaqirish
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# 2️⃣ DBSCAN proyekt classi
class DBSCANProject:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.model = None
        self.data = None
        self.labels = None

    def load_and_filter_data(self):
        # 🌸 Iris ma’lumotlarini yuklash
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)

        # Filtrlash: sepal length > 4.5 bo‘lganlarni olish
        filtered_df = df[df[iris.feature_names[0]] > 4.5].copy()
        self.data = filtered_df
        print("🔎 Filtrlangan ma’lumotlar:", self.data.shape)

    def preprocess(self):
        # Standartlashtirish
        self.data_scaled = self.scaler.fit_transform(self.data)

    def build_and_fit(self):
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = self.model.fit_predict(self.data_scaled)
        unique_clusters = set(self.labels)
        print(f"✅ DBSCAN bajarildi. Topilgan klasterlar: {unique_clusters}")

    def visualize_clusters(self):
        # PCA orqali 2D vizualizatsiya
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.data_scaled)

        plt.figure(figsize=(6,5))
        plt.scatter(
            reduced[:,0],
            reduced[:,1],
            c=self.labels,
            cmap='plasma',
            edgecolors='k'
        )
        plt.title(f"DBSCAN Klasterlash (eps={self.eps}, min_samples={self.min_samples})")
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.show()

    def cluster_summary(self):
        clustered_df = self.data.copy()
        clustered_df['Cluster'] = self.labels
        summary = clustered_df.groupby('Cluster').mean()
        print("\n📊 Klaster xulosalari (o‘rtacha qiymatlar):")
        print(summary)

# 3️⃣ Loyihani ishga tushirish
project = DBSCANProject(eps=0.7, min_samples=5)  # eps va min_samples parametrlarini sozlashingiz mumkin
project.load_and_filter_data()    # Ma’lumotni yuklash va filtrlash
project.preprocess()              # Standartlashtirish
project.build_and_fit()           # DBSCAN modelini ishga tushirish
project.visualize_clusters()      # Klasterlarni chizish
project.cluster_summary()         # Klaster xulosalari
