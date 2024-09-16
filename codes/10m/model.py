import torch.nn as nn

# Modèle personnalisé utilisant un réseau de neurones simple
class WindSpeedPredictor(nn.Module):
    def __init__(self):
        super(WindSpeedPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 64),  # 10 car il y a 10 features d'entrée (excluant SpeedAvg)
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout avec 40% de probabilité
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Prédire SpeedAvg
        )

    def forward(self, x):
        return self.fc(x)

