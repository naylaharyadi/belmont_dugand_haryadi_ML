import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Chargement des données
data = pd.read_csv('bddclean.csv')

# Grouper les données par filière et calculer le total pour chaque type de mention
grouped_data = data.groupby('Filière de formation').agg({
    'Dont effectif des admis néo bacheliers avec mention Assez Bien au bac': 'sum',
    'Dont effectif des admis néo bacheliers avec mention Bien au bac': 'sum',
    'Dont effectif des admis néo bacheliers avec mention Très Bien au bac': 'sum'
}).reset_index()

# Préparation des données pour la visualisation
filières = grouped_data['Filière de formation']
mentions_assez_bien = grouped_data['Dont effectif des admis néo bacheliers avec mention Assez Bien au bac']
mentions_bien = grouped_data['Dont effectif des admis néo bacheliers avec mention Bien au bac']
mentions_très_bien = grouped_data['Dont effectif des admis néo bacheliers avec mention Très Bien au bac']

# Création du graphique à barres
bar_width = 0.25  # Largeur des barres
index = np.arange(len(filières))  # L'index pour le placement des barres

plt.figure(figsize=(14, 8))  # Taille du graphique
plt.bar(index, mentions_assez_bien, bar_width, label='Assez Bien')
plt.bar(index + bar_width, mentions_bien, bar_width, label='Bien')
plt.bar(index + 2 * bar_width, mentions_très_bien, bar_width, label='Très Bien')

plt.xlabel('Filière de formation')
plt.ylabel('Nombre de candidats admis')
plt.title('Impact de la mention au baccalauréat sur le choix des filières')
plt.xticks(index + bar_width, filières, rotation=90)
plt.legend()

plt.tight_layout()
plt.show()
