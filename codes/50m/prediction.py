import torch
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from model import WindSpeedPredictor 

# Dataset personnalisé pour la prédiction
class PredictionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Charger les données pour la prédiction
data_pred = pd.read_csv('data/New true Datasets 10-09-2024/test datasets/test 50m full.csv')

# Séparer les caractéristiques d'entrée
features_pred = data_pred[['Speed Avg 10m', 'SpeedMax', 'DirectionAvg', 'TemperatureAvg', 'TemperatureMax',
                           'PressureAvg', 'PressureMax', 'HumidtyAvg', 'HumityMax', 'height']].values

# Charger le scaler sauvegardé
scaler = joblib.load('Code/50m/model/scaler.joblib')

# Normalisation des données
features_pred_scaled = scaler.transform(features_pred)

# Créer le dataset pour la prédiction et le DataLoader
prediction_dataset = PredictionDataset(features_pred_scaled)
prediction_loader = DataLoader(prediction_dataset, batch_size=1, shuffle=False)

# Initialiser le modèle
model = WindSpeedPredictor()
model.load_state_dict(torch.load('Code/50m/model/wind_speed_predictor.pth'))
model.eval()

# Faire des prédictions
predictions = []
with torch.no_grad():
    for batch_data in prediction_loader:
        pred = model(batch_data).squeeze()
        predictions.append(pred.item())

# Ajouter les prédictions aux données originales
data_pred['PredictedSpeedAvg'] = predictions

# Enregistrer les résultats dans un fichier CSV
data_pred.to_csv('Code/50m/prediction/predictions 50m full.csv', index=False)

print("Prédictions sauvegardées dans 'Code/50m/prediction/predictions 50m full.csv'.")
