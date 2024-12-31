import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les deux bases de données
admissions_cleaned = pd.read_csv("admissions_cleaned.csv")
socio_grouped = pd.read_csv("socio_grouped.csv")

# S'assurer que "Region Code" est de type string pour la fusion
admissions_cleaned["Region Code"] = admissions_cleaned["Region Code"].astype(str)
socio_grouped["Region Code"] = socio_grouped["Region Code"].astype(str)

# Fusionner les deux bases sur "Region Code"
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

# Créer une variable cible catégorielle pour les taux d'admission (Low, Medium, High)
merged_data["Admission Category"] = pd.qcut(
    merged_data["Admission Rate"], 
    q=3, 
    labels=["Low", "Medium", "High"]
)

# Préparer les données pour la classification
X = merged_data[["Median Income", "Low Income Rate", "Gini Index"]]
y = merged_data["Admission Category"]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner un modèle de forêt aléatoire
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Importance des caractéristiques
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Visualisation de l'importance des caractéristiques
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance, y=feature_importance.index, palette="viridis")
plt.title("Importance des caractéristiques dans le modèle Forêt aléatoire")
plt.xlabel("Score d'importance")
plt.ylabel("Caractéristiques")
plt.grid(alpha=0.3)
plt.show()
