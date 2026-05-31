import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample data
data = {
    'Annual Income (k$)': [15, 16, 17, 18, 25, 30, 35, 40, 45, 50,
                           55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    'Spending Score (1-100)': [39, 81, 6, 77, 40, 76, 6, 94, 3, 72,
                               14, 99, 15, 77, 13, 90, 20, 88, 25, 95]
}

df = pd.DataFrame(data)

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# K-Means with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Plot clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200, marker='X')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
