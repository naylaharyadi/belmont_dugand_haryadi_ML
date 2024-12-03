import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données
data_path = "bddclean.csv"
data = pd.read_csv(data_path)

# Préparation des données
# Assurer que toutes les données sont numériques et gérer les valeurs manquantes
data['Effectif total des candidats pour une formation'] = pd.to_numeric(data['Effectif total des candidats pour une formation'], errors='coerce')
data['% d’admis dont filles'] = pd.to_numeric(data['% d’admis dont filles'], errors='coerce')
data = data.dropna(subset=['Effectif total des candidats pour une formation', '% d’admis dont filles'])

# Sélectionner les colonnes pour la régression
X = data[['Effectif total des candidats pour une formation']]
y = data['% d’admis dont filles']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Calcul des métriques de performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Affichage des résultats
print(f'Coefficient: {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')
print(f'RMSE: {rmse}')
print(f'R^2: {r2}')

# Visualisation des résultats
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Données réelles')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Prédiction du modèle')
plt.title('Régression Linéaire pour Prédire le % d’admis dont filles')
plt.xlabel('Effectif total des candidats pour une formation')
plt.ylabel('% d’admis dont filles')
plt.legend()
plt.show()
