# Wind Speed Prediction

This project uses approaches for wind speed prediction: LLM fine-tuning and a neural network for regression. The aim is to predict average wind speed using meteorological parameters.

```bash
Project/
│
├── Codes/
│ ├── train_bert.py # BERT model training script
│ ├── models/ # Directory of trained models
│ ├── 10m/ # Scripts specific to 10m data
│ ├── 50m/ # Scripts specific to data at 50m
│ └── 100m/ # Scripts specific to data at 100m
│
├── datasets/ # Real datasets for training and testing
├── requirements.txt # This file contains all the packages to be installed.
└── README.md # This file
```

# Prerequisites

Before you start, make sure you have installed the following:

Python 3.8+
pip for dependency management

# Installation

Install the necessary libraries using the requirements.txt file:

```bash
pip install -r requirements.txt
```

# Usage

## BERT model training

The train_bert.py script trains a finetuned BERT model on weather data.

```bash
python Code/train_bert.py
```

## Training the neural network

The main.py script is used to train a neural network to predict the average wind speed at a given height.

```bash
python Code/50m/main.py
```

# Process:

1. Loading weather data from a CSV file.
2. Data normalization using StandardScaler.
3. Training the neural network for a regression task.
4. Calculate MAE and R2-score.
5. Save model and scaler in Code/50m/model/.

## Neural Network predictions

Use predict_nn.py to generate predictions with the neural network model.

```bash
python Code/50m/prediction.py
```

# Results

During training and testing, model performance logs are displayed in the console.

# Contact us

If you have any questions or suggestions, please don't hesitate to contact us.
