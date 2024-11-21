
# Prédiction des Rendements Mensuels d'AAPL avec un Modèle LSTM

Ce projet utilise un modèle LSTM (Long Short-Term Memory) pour prédire les rendements mensuels de l'action AAPL à partir de données historiques. L'objectif est d'anticiper les performances futures à partir des tendances passées.

## Objectif

Le projet vise à :
1. **Analyser les rendements historiques** : Visualiser les données temporelles pour identifier les tendances et les fluctuations.
2. **Entraîner un modèle prédictif** : Utiliser un réseau de neurones récurrent (LSTM) pour apprendre des données passées et prévoir les rendements futurs.
3. **Générer des prédictions** : Fournir des estimations des rendements pour les mois à venir.

## Données

Le projet se base sur un fichier CSV contenant les données mensuelles suivantes :
- **Year-Month** : Année et mois du relevé.
- **return** : Rendement mensuel de l'action.
- **start_price, end_price, total_dividends** : Non utilisées dans ce projet, mais présentes pour information.

Exemple de données :
```csv
Year-Month,return
1980-12,-8.59
1981-01,-36.94
1981-02,-23.71
...
```

## Méthodologie

1. **Préparation des Données** :
   - Conversion de la colonne `Year-Month` en type datetime.
   - Création d'une fenêtre glissante (lookback) pour capturer les rendements des 6 derniers mois comme variables explicatives.

2. **Normalisation** :
   - Les rendements sont normalisés entre -1 et 1 pour faciliter l'entraînement du modèle.

3. **Modèle** :
   - Utilisation d'un modèle LSTM, adapté aux données temporelles, avec :
     - **Input** : Rendements des 6 derniers mois.
     - **Output** : Prédiction du rendement du mois suivant.

4. **Entraînement et Validation** :
   - Division des données en ensembles d'entraînement (95%) et de test (5%).
   - Calcul de la perte avec une fonction d'erreur quadratique moyenne (MSELoss).

5. **Prédictions Futures** :
   - Le modèle prédit le rendement d'un mois suivant les dernières données connues.

## Résultats

Le modèle génère des prévisions pour les rendements futurs. Un graphique est produit pour comparer les rendements historiques aux prédictions futures.

### Exemple de Résultat
```
Year-Month        Predicted Return
2024-11-01        2.788696
```

## Fichiers du Projet

- **Code Python** : 
  - Préparation des données, entraînement du modèle et génération des prédictions.
- **Fichier CSV** : Contient les données historiques (`monthly_aapl_data.csv`).
- **Modèle Sauvegardé** : Modèle LSTM (`lstm_model.pth`).

## Dépendances

- Python 3.x
- Bibliothèques : 
  - `pandas`
  - `numpy`
  - `torch`
  - `matplotlib`

## Instructions pour l'Exécution

1. Cloner le dépôt ou télécharger les fichiers du projet.
2. Installer les dépendances avec `pip install -r requirements.txt`.
3. Placer le fichier `monthly_aapl_data.csv` dans le répertoire du projet.
4. Exécuter le script principal pour entraîner le modèle et générer les prédictions :
   ```bash
   python main.py
   ```
5. Visualiser les résultats dans le graphique produit.

## Limitations et Perspectives

### Limitations :
- Le modèle est basé uniquement sur les données passées et peut ne pas capturer les événements imprévus affectant les marchés financiers.
- La précision dépend de la qualité et de la quantité des données disponibles.

### Perspectives :
- Ajouter des variables explicatives comme les indicateurs macroéconomiques.
- Tester d'autres architectures de modèles (GRU, Transformers).
- Améliorer la gestion des hyperparamètres pour optimiser les performances.

## Auteur

Ce projet a été conçu pour démontrer l'application des modèles LSTM dans les séries temporelles financières.
