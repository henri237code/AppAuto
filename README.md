# AppAuto
Apprentissage Automatique


Ce projet vise à développer une application web qui fournit une interface interactive pour 
permettre aux utilisateurs de télécharger un jeu de données et de créer des modèles 
d'apprentissage automatique pour des tâches de classification ou de régression. L'application 
inclura les fonctionnalités suivantes : 
● Fonctionnalité de téléchargement de fichier 
● Entraînement et évaluation de modèles de classification (4 modèles) 
● Entraînement et évaluation de modèles de régression (4 modèles) 
● Entraînement et évaluation de modèles d’apprentissage automatique avec la 
normalisation : StandardScaler, Normalizer et MinMaxSacler 
● Interface pour faire des prédictions avec les modèles entraînés 
2. Fonctionnalités 
1. Téléchargement de Fichier : 
○ Permettre aux utilisateurs de télécharger des fichiers CSV. 
○ Afficher un signal de chargement pendant le traitement du fichier. 
2. Onglet d'Apprentissage Automatique (ML) : 
○ Permettre aux utilisateurs de spécifier si le problème est une tâche de 
classification ou de régression. 
○ Permettre aux utilisateurs de sélectionner la variable cible (output). 
○ Entraîner 4 algorithmes différents et afficher leurs métriques de performance dans 
un tableau. 
○ Pour la classification : Précision, Rappel, Précision, F1-score. 
○ Pour la régression : MAE, MSE, R^2. 
○ Afficher un signal de chargement pendant le traitement. 
3. Interface de Prédiction : 
○ Fournir une interface permettant aux utilisateurs de saisir de nouvelles données. 
○ Permettre aux utilisateurs de sélectionner un modèle entraîné spécifique pour faire 
des prédictions. 
3. Mise en Œuvre 
1. Configuration et Dépendances : 
○ Installer les bibliothèques nécessaires (Streamlit, Pandas, Scikit-learn, plotly etc.). 
2. Fonctionnalité de Téléchargement de Fichier : 
○ Créer une fonction pour gérer les téléchargements de fichiers. 
○ Afficher le contenu du fichier téléchargé. 
3. Module ML : 
○ Créer des fonctions pour le prétraitement des données (encodage des variables 
catégorielles). 
○ Créer des fonctions pour la formation des différents modèles d'apprentissage 
automatique. 
○ Créer des fonctions pour évaluer les modèles. 
○ Afficher les métriques de performance des modèles. 
4. Interface de Prédiction : 
○ Créer des fonctions pour permettre aux utilisateurs de saisir de nouvelles données. 
○ Créer des fonctions pour faire des prédictions en utilisant les modèles entraînés. 
5. Disposition et Navigation de l'Application : 
○ Utiliser des onglets pour organiser l'application. 
○ Créer le layout principal de l'application. 