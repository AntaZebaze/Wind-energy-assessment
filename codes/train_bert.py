import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader, Dataset
import joblib


# Dataset personnalisé
class WindSpeedDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Modèle personnalisé utilisant BERT
class BERTWindSpeedPredictor(nn.Module):
    def __init__(self):
        super(BERTWindSpeedPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 64),  # 9 car il y a 16 features d'entrée
            nn.ReLU(),
            nn.Dropout(0.4), # Dropout avec 30% de probabilite
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Prédire SpeedAvg
        )

    def forward(self, x):
        return self.fc(x)

# Chargement des données d'entraînement
data = np.loadtxt('data/New true Datasets 10-09-2024/Datasets/train_mix.csv', delimiter=',', skiprows=1, usecols=range(2, 11)) 
labels = np.loadtxt('data/New true Datasets 10-09-2024/Datasets/train_mix.csv', delimiter=',', skiprows=1, usecols=1)

# Normalisation des données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Créer le dataset et DataLoader
dataset = WindSpeedDataset(data_scaled, labels)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialiser le modèle
model = BERTWindSpeedPredictor()
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

# Sauvegarder le modèle et le scaler
torch.save(model.state_dict(), 'models/train_on_train_mix/bert_wind_speed_predictor_new.pth')
# torch.save(scaler, 'models/trained_on_8000_data/scaler.pth')
joblib.dump(scaler, 'models/train_on_train_mix/scaler_new.joblib')
print("Modèle et scaler sauvegardés.")
