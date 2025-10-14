# Tsunami Prediction System 🌊

A machine learning system to predict tsunami severity and intensity based on earthquake and geological data.

## Files

- `model.py` - Neural network model definition
- `train.py` - Training script to train the models
- `predict.py` - Prediction wrapper class
- `app.py` - Streamlit web application
- `tsunami.csv` - Dataset (required)
- `requirements.txt` - Python dependencies

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the models:

```bash
python train.py
```

This will create `tsunami_models.pth` file with trained models.

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Usage

The web app allows you to:

- Input earthquake magnitude, location, and water height
- Set tsunami intensity and other parameters
- Get predictions for severity (Minor/Moderate/Major/Extreme)
- Get predictions for intensity (Low/High)
- View confidence scores for predictions

## Model Architecture

- Simple feedforward neural network with 2 hidden layers (64, 32 neurons)
- Two separate models: one for severity (4 classes), one for intensity (2 classes)
- Uses BatchNorm and Dropout for regularization
- Trained with early stopping and learning rate scheduling

## Dataset

The model is trained on historical tsunami data with features:

- Year, Month, Day, Hour, Minute, Second
- Tsunami Event Validity, Tsunami Cause Code
- Earthquake Magnitude, Volcanic Activity
- Latitude, Longitude
- Maximum Water Height, Number of Runups
- Tsunami Magnitude (Abe), Tsunami Magnitude (Iida)
- Tsunami Intensity
