import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les bases nettoyées
admissions_cleaned = pd.read_csv("admissions_cleaned.csv")
socio_grouped = pd.read_csv("socio_grouped.csv")

# Assurez-vous que 'Region Code' est au même format dans les deux datasets
admissions_cleaned["Region Code"] = admissions_cleaned["Region Code"].astype(str)
socio_grouped["Region Code"] = socio_grouped["Region Code"].astype(str)

# Fusionner les deux bases
merged_data = pd.merge(
    admissions_cleaned,
    socio_grouped,
    on="Region Code",
    how="inner"
)

# Ajouter une colonne pour le taux d'admission
merged_data["Admission Rate"] = (
    merged_data["Effectif des admis néo bacheliers"]
    / merged_data["Effectif total des candidats pour une formation"]
)

# Normalisation des données pour le clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(
    merged_data[["Median Income", "Low Income Rate", "Gini Index", "Admission Rate"]]
)

# Application du clustering K-means
kmeans = KMeans(n_clusters=3, random_state=42)
merged_data["Cluster"] = kmeans.fit_predict(scaled_features)

# Visualisation des clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=merged_data["Median Income"],
    y=merged_data["Admission Rate"],
    hue=merged_data["Cluster"],
    palette="viridis",
    style=merged_data["Cluster"],
    s=100
)
plt.title("Clustering Regions Based on Socio-economic Indicators and Admission Rates")
plt.xlabel("Median Income")
plt.ylabel("Admission Rate")
plt.legend(title="Cluster")
plt.show()
