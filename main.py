import yfinance as yf
import pandas as pd

# Définir le ticker
ticker = 'AAPL'

# Télécharger les données historiques
data = yf.download(ticker, period='max', actions=True, auto_adjust=False)

# Aplatir les colonnes du MultiIndex
data.columns = [f"{col[0]}" for col in data.columns]

# Vérifier les colonnes disponibles après aplatissement
print("Colonnes disponibles après aplatissement :", data.columns)

# Convertir l'index en datetime pour faciliter les manipulations
data.index = pd.to_datetime(data.index)

# Ajouter une colonne pour l'année
data['Year-Month'] = data.index.to_period('M').astype('str')

# Grouper par année pour calculer les statistiques nécessaires
monthly_data = data.groupby('Year-Month').agg(
    start_price=('Open', 'first'),  # Premier prix de l'année
    end_price=('Adj Close', 'last'),   # Dernier prix de l'année
    total_dividends=('Dividends', 'sum')  # Somme des dividendes de l'année
)

# Calculer le rendement
monthly_data['return'] = (((monthly_data['end_price'] - monthly_data['start_price']) + monthly_data['total_dividends']) / monthly_data['start_price']) * 100

# Réinitialiser l'index pour que l'année devienne une colonne
monthly_data.reset_index(inplace=True)

# Sauvegarder les données dans un fichier CSV
monthly_data.to_csv('annual_aapl_data.csv', index=False)

# Afficher un aperçu des données
print(monthly_data)