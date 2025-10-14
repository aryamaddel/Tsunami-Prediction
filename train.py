import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from model import TsunamiNet
import warnings

warnings.filterwarnings("ignore")


class TsunamiDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def classify_tsunami_severity(row):
    water_height = row.get("Maximum Water Height (m)", 0)
    intensity = row.get("Tsunami Intensity", 0)
    water_height = 0 if pd.isna(water_height) else water_height
    intensity = 0 if pd.isna(intensity) else intensity

    if water_height >= 10 or intensity >= 5:
        return 3
    elif water_height >= 5 or intensity >= 4:
        return 2
    elif water_height >= 2 or intensity >= 3:
        return 1
    else:
        return 0


def train_model(
    model, train_loader, test_loader, num_epochs=100, learning_rate=0.001, task_name=""
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    best_accuracy = 0.0
    patience = 15
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)
                _, predicted = torch.max(model(features), 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        test_accuracy = 100 * correct / total

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), f"{task_name}_model.pth")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Acc: {test_accuracy:.2f}%"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(f"{task_name}_model.pth"))
    return model, best_accuracy


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_csv("tsunami.csv")
    df_processed = df.copy()
    df_processed["Year"] = pd.to_numeric(df_processed["Year"], errors="coerce")
    df_processed = df_processed[df_processed["Year"] >= 1900].copy()

    df_processed["Tsunami_Severity"] = df_processed.apply(
        classify_tsunami_severity, axis=1
    )
    df_processed["Is_High_Intensity"] = (
        (df_processed["Maximum Water Height (m)"].fillna(0) >= 2)
        | (df_processed["Tsunami Intensity"].fillna(0) >= 3)
    ).astype(int)

    input_features = [
        "Year",
        "Mo",
        "Dy",
        "Hr",
        "Mn",
        "Sec",
        "Tsunami Event Validity",
        "Tsunami Cause Code",
        "Earthquake Magnitude",
        "Vol",
        "Latitude",
        "Longitude",
        "Maximum Water Height (m)",
        "Number of Runups",
        "Tsunami Magnitude (Abe)",
        "Tsunami Magnitude (Iida)",
        "Tsunami Intensity",
    ]
    available_input_features = [
        col for col in input_features if col in df_processed.columns
    ]

    X = df_processed[available_input_features].copy()
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    y_severity = df_processed["Tsunami_Severity"].values
    y_intensity = df_processed["Is_High_Intensity"].values

    X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
        X_scaled, y_severity, test_size=0.2, random_state=42, stratify=y_severity
    )
    X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(
        X_scaled, y_intensity, test_size=0.2, random_state=42, stratify=y_intensity
    )

    train_dataset_sev = TsunamiDataset(X_train_sev, y_train_sev)
    test_dataset_sev = TsunamiDataset(X_test_sev, y_test_sev)
    train_dataset_int = TsunamiDataset(X_train_int, y_train_int)
    test_dataset_int = TsunamiDataset(X_test_int, y_test_int)

    batch_size = 32
    train_loader_sev = DataLoader(
        train_dataset_sev, batch_size=batch_size, shuffle=True
    )
    test_loader_sev = DataLoader(test_dataset_sev, batch_size=batch_size, shuffle=False)
    train_loader_int = DataLoader(
        train_dataset_int, batch_size=batch_size, shuffle=True
    )
    test_loader_int = DataLoader(test_dataset_int, batch_size=batch_size, shuffle=False)

    
    input_size = X_scaled.shape[1]
    model_severity = TsunamiNet(input_size, [64, 32], 4, 0.3)
    model_intensity = TsunamiNet(input_size, [64, 32], 2, 0.3)
    
    print("Training Severity Classification Model")
    model_severity, sev_best = train_model(
        model_severity,
        train_loader_sev,
        test_loader_sev,
        num_epochs=100,
        learning_rate=0.001,
        task_name="severity",
    )
    print(f"Severity Model Best Accuracy: {sev_best:.2f}%")

    print("\nTraining Intensity Classification Model")
    model_intensity, int_best = train_model(
        model_intensity,
        train_loader_int,
        test_loader_int,
        num_epochs=100,
        learning_rate=0.001,
        task_name="intensity",
    )
    print(f"Intensity Model Best Accuracy: {int_best:.2f}%")

    torch.save(
        {
            "severity_model": model_severity.state_dict(),
            "intensity_model": model_intensity.state_dict(),
            "scaler": scaler,
            "imputer": imputer,
            "feature_names": available_input_features,
            "severity_classes": ["Minor", "Moderate", "Major", "Extreme"],
            "intensity_classes": ["Low Intensity", "High Intensity"],
            "input_size": input_size,
        },
        "tsunami_models.pth",
    )

    print("\nModels saved to 'tsunami_models.pth'")
