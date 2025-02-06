import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the data
car_pca_df = pd.read_csv("car_pca_df.csv")

# Print column names to check for the target variable
print(car_pca_df.columns)

# Remove target variable
if "IsBadBuy" in car_pca_df.columns:
    cluster_df = car_pca_df.drop(columns=["IsBadBuy"])
else:
    print("The column 'IsBadBuy' is not found in the DataFrame.")

# Proceed with the rest of the code only if the target variable is found
if "IsBadBuy" in car_pca_df.columns:
    # Search for the best K
    def find_K(K, data):
        kmeans = KMeans(n_clusters=K, n_init=20, random_state=15)
        kmeans.fit(data)
        return kmeans.inertia_

    k_val = list(range(1, 9))
    find_K_val = [find_K(k, cluster_df) for k in k_val]

    # Elbow Curve
    plt.figure(figsize=(8, 6))
    plt.plot(k_val, find_K_val, marker='o')
    plt.xlabel("Number of clusters K")
    plt.ylabel("Total within-clusters sum of squares")
    plt.title("Elbow Method For Optimal K")
    plt.grid(True)
    plt.show()

    # Model with best K
    kmeans_car = KMeans(n_clusters=4, n_init=20, max_iter=100, random_state=15, algorithm="lloyd")
    kmeans_car.fit(cluster_df)

    # Clusters 2D representation
    pca = PCA(2)
    plot_columns = pca.fit_transform(cluster_df)
    plt.scatter(plot_columns[:, 0], plot_columns[:, 1], c=kmeans_car.labels_, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Cluster Representation in 2D')
    plt.colorbar()
    plt.show()
