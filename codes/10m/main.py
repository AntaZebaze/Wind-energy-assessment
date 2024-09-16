import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import joblib
from model import WindSpeedPredictor

# Dataset personnalisé
class WindSpeedDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Chargement des données
df = pd.read_csv('data/New true Datasets 10-09-2024/Train Datasets/Dataset 10m.csv')

# Séparer les caractéristiques d'entrée et la cible (SpeedAvg)
features = df[['SpeedMax', 'DirectionAvg', 'TemperatureAvg', 'TemperatureMax', 
               'PressureAvg', 'PressureMax', 'HumidtyAvg', 'HumityMax', 'height']].values
labels = df['SpeedAvg'].values

# Normalisation des données
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Créer le dataset et DataLoader
dataset = WindSpeedDataset(features_scaled, labels)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialiser le modèle
model = WindSpeedPredictor()
optimizer = optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()

# Entraînement
model.train()
for epoch in range(50):  # Nombre d'époques
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_data).squeeze()
        loss = criterion(predictions, batch_labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Évaluation du modèle
model.eval()
with torch.no_grad():
    # Prédictions sur l'ensemble des données
    predictions = []
    true_labels = []
    for batch_data, batch_labels in DataLoader(dataset, batch_size=32, shuffle=False):
        batch_predictions = model(batch_data).squeeze()
        predictions.extend(batch_predictions.numpy())
        true_labels.extend(batch_labels.numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Calcul des métriques
    mae = mean_absolute_error(true_labels, predictions)
    r2 = r2_score(true_labels, predictions)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R2 Score: {r2}")

# Sauvegarder le modèle et le scaler
torch.save(model.state_dict(), 'Code/10m/model/wind_speed_predictor2.pth')
joblib.dump(scaler, 'Code/10m/model/scaler2.joblib')
print("Modèle et scaler sauvegardés.")
