import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
import seaborn as sns 


base_path = os.path.dirname(__file__)  
file_name = "Filedata Data Jumlah Penduduk Provinsi DKI Jakarta Berdasarkan Agama.csv"
file_path = os.path.join(base_path, file_name)
df = pd.read_csv(file_path)


#Hapus periode_data
df = df.drop(['periode_data'], axis=1)


#Label Encoding
encoder = LabelEncoder()
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = encoder.fit_transform(df[col]) 
X = df.drop('agama', axis=1)
y = df['agama']


#PCA
scaler = StandardScaler()
scaler.fit(X)

scaled_data = scaler.transform(X)

features = scaled_data.T
cov_matrix = np.cov(features)

values, vectors = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(values)[::-1]

values = values[sorted_indices]
vectors = vectors[:, sorted_indices]

explained_variances = []

for i in range(len(values)):
    explained_variances.append(values[i] / np.sum(values) * 100)

print(np.sum(explained_variances))
print(explained_variances)

feature_names = np.array(X.columns)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
component_weights = pca.components_

feature_weights_mapping = {}

for i, component in enumerate(component_weights):
    component_feature_weights = zip(feature_names, component)
    feature_weights_mapping[f"Component {i+1}"] = sorted(
      component_feature_weights, key=lambda x: abs(x[1]), reverse=True)

print("Feature names contributing to Component 1:")
for feature, weight in feature_weights_mapping["Component 1"]:
    print(f"{feature}: {weight}")

print("\nFeature names contributing to Component 2:")
for feature, weight in feature_weights_mapping["Component 2"]:
    print(f"{feature}: {weight}")

print("\nFeature names contributing to Component 3:")
for feature, weight in feature_weights_mapping["Component 3"]:
    print(f"{feature}: {weight}")


#Elbow Method
inertia = []
for cluster in range(1, 6):
    km = KMeans(n_clusters=cluster, n_init=10)
    km = km.fit(df)
    inertia.append(km.inertia_)

plt.plot(range(1, 6), inertia, 'bx-')
plt.xlabel('Jumlah klaster')
plt.ylabel('Inersia')
plt.title('Visualisasi elbow method')
plt.show()


#Silhouette Coefficient
print("\nPrint silhouette coefficient:")
for k in [2, 3, 4, 5]:
    clusterer = KMeans(n_clusters = k, n_init=10)
    cluster_labels = clusterer.fit_predict(df)
    silhouette_avg = silhouette_score(df, cluster_labels)
    print(f"Untuk k = {k}, rata-rata silhouette_coefficient adalah: {silhouette_avg}")

fig, ax = plt.subplots(2, 2, figsize=(15,12))
fig.suptitle("Visualisasi Silhouette Coefficient untuk beberapa nilai k")

for k in [2, 3, 4, 5]:
    clusterer = KMeans(n_clusters=k, n_init=10)

    q, mod = divmod(k, 2)
    ax[q-1][mod].set_title(f"k = {k}")
    visualizer = SilhouetteVisualizer(clusterer, ax = ax[q-1][mod])
    visualizer.fit(df)
plt.show()


#Agglomerative Clustering
kmeans = KMeans(n_clusters=2, n_init=10)
assignment = kmeans.fit_predict(df)

df_with_clusters = pd.DataFrame(df.copy())
df_with_clusters['agama'] = assignment
df_with_clusters.head()

fig = plt.figure(figsize = (10, 10))
ax = plt.axes(projection="3d")

x = df_with_clusters['wilayah']
y = df_with_clusters['kecamatan']
z = df_with_clusters['kelurahan']
cluster = df_with_clusters['agama']

ax.scatter(x, y, z, c = cluster, cmap = "rainbow")
plt.title("Klaster Data Pelanggan Supermarket")
plt.grid(False)
ax.set_xlabel('wilayah')
ax.set_ylabel('kecamatan')
ax.set_zlabel('kelurahan')
plt.show()
