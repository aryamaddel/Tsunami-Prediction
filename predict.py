import torch
import torch.nn.functional as F
import numpy as np
from model import TsunamiNet


class TsunamiPredictor:
    def __init__(self, model_path="tsunami_models.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        self.scaler = checkpoint["scaler"]
        self.imputer = checkpoint["imputer"]
        self.feature_names = checkpoint["feature_names"]
        self.severity_classes = checkpoint["severity_classes"]
        self.intensity_classes = checkpoint["intensity_classes"]
        self.input_size = checkpoint["input_size"]

        self.model_severity = TsunamiNet(self.input_size, [64, 32], 4, 0.3)
        self.model_intensity = TsunamiNet(self.input_size, [64, 32], 2, 0.3)

        self.model_severity.load_state_dict(checkpoint["severity_model"])
        self.model_intensity.load_state_dict(checkpoint["intensity_model"])

        self.model_severity.to(self.device)
        self.model_intensity.to(self.device)

        self.model_severity.eval()
        self.model_intensity.eval()

    def predict(
        self,
        earthquake_magnitude,
        latitude,
        longitude,
        earthquake_depth=10.0,
        year=None,
        month=None,
        day=None,
        hour=0,
        minute=0,
        second=0,
        event_validity=4,
        cause_code=1,
        vol=0,
    ):
        """
        Predict tsunami characteristics based on earthquake parameters only.

        Args:
            earthquake_magnitude: Magnitude of the earthquake (Richter scale)
            latitude: Latitude of earthquake epicenter
            longitude: Longitude of earthquake epicenter
            earthquake_depth: Depth of earthquake in km (default: 10.0)
            year, month, day: Date of event (defaults to current date if None)
            event_validity: Validity code (1-4, default: 4)
            cause_code: Cause of tsunami (1=Earthquake, 2=Volcanic, etc.)
            vol: Volcanic explosivity index (0 if not volcanic)

        Returns:
            Dictionary with severity and intensity predictions
        """
        from datetime import datetime

        # Use current date if not provided
        if year is None or month is None or day is None:
            now = datetime.now()
            year = year or now.year
            month = month or now.month
            day = day or now.day

        # Build input data with earthquake parameters only
        # Tsunami parameters are set to 0 (unknown) as they are what we're predicting
        input_data = [
            year,
            month,
            day,
            hour,
            minute,
            second,
            event_validity,
            cause_code,
            earthquake_magnitude,
            vol,
            latitude,
            longitude,
        ]
        # Set tsunami parameters to 0 (unknown/to be predicted)
        input_data.append(0)  # water_height (unknown - to be predicted)
        input_data.append(0)  # num_runups (unknown - to be predicted)
        input_data.append(0)  # tsunami_magnitude_abe (unknown - to be predicted)
        input_data.append(0)  # tsunami_magnitude_iida (unknown - to be predicted)
        input_data.append(0)  # intensity (unknown - to be predicted)

        while len(input_data) < self.input_size:
            input_data.append(0)
        input_data = input_data[: self.input_size]

        input_tensor = torch.FloatTensor(
            self.scaler.transform(
                self.imputer.transform(np.array(input_data).reshape(1, -1))
            )
        ).to(self.device)

        with torch.no_grad():
            severity_probs = F.softmax(self.model_severity(input_tensor), dim=1)
            intensity_probs = F.softmax(self.model_intensity(input_tensor), dim=1)
            severity_pred = torch.argmax(severity_probs, dim=1).cpu().numpy()[0]
            intensity_pred = torch.argmax(intensity_probs, dim=1).cpu().numpy()[0]

        return {
            "severity": self.severity_classes[severity_pred],
            "severity_confidence": f"{severity_probs.max().cpu().numpy():.2%}",
            "intensity": self.intensity_classes[intensity_pred],
            "intensity_confidence": f"{intensity_probs.max().cpu().numpy():.2%}",
        }
