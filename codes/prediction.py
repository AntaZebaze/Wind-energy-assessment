import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel
import joblib

# Dataset personnalisé pour les prédictions
class WindSpeedPredictionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Modèle personnalisé utilisant BERT (même architecture que pour l'entraînement)
class BERTWindSpeedPredictor(nn.Module):
    def __init__(self):
        super(BERTWindSpeedPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(16, 64),  # 16 car il y a 16 features d'entrée
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Prédire SpeedAvg
        )

    def forward(self, x):
        return self.fc(x)

# Charger le modèle et le scaler sauvegardés
model = BERTWindSpeedPredictor()
model.load_state_dict(torch.load('models/trained_on_8000_data/bert_wind_speed_predictor.pth', map_location=torch.device('cpu')))
model.eval()

scaler = joblib.load('models/trained_on_8000_data/scaler.joblib')

# Charger les données de test
test_data = pd.read_csv('data/test_data_1.csv', delimiter=',', usecols=range(2, 18)).values
test_data_scaled = scaler.transform(test_data)

# Créer le dataset et DataLoader pour les données de test
test_dataset = WindSpeedPredictionDataset(test_data_scaled)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Effectuer les prédictions
predictions = []
with torch.no_grad():
    for batch_data in test_loader:
        batch_predictions = model(batch_data).squeeze().numpy()
        predictions.extend(batch_predictions)

# Convertir les résultats en tableau NumPy et les ajouter au DataFrame
predictions = np.array(predictions)
test_df = pd.read_csv('data/New true Datasets 10-09-2024/Datasets/test_mix_prediction.csv')
test_df['PredictedSpeedAvg'] = predictions

# Sauvegarder les prédictions dans un fichier CSV
test_df.to_csv('predictions.csv', index=False)
print("Prédictions sauvegardées dans 'predictions.csv'.")
