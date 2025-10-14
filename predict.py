import torch
import torch.nn.functional as F
import numpy as np
from model import TsunamiPredictor

class TsunamiPredictor:
    def __init__(self, model_path="tsunami_models.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.scaler = checkpoint["scaler"]
        self.imputer = checkpoint["imputer"]
        self.feature_names = checkpoint["feature_names"]
        self.severity_classes = checkpoint["severity_classes"]
        self.intensity_classes = checkpoint["intensity_classes"]
        self.input_size = checkpoint["input_size"]
        
        from model import TsunamiPredictor as Model
        self.model_severity = Model(self.input_size, [64, 32], 4, 0.3)
        self.model_intensity = Model(self.input_size, [64, 32], 2, 0.3)
        
        self.model_severity.load_state_dict(checkpoint["severity_model"])
        self.model_intensity.load_state_dict(checkpoint["intensity_model"])
        
        self.model_severity.to(self.device)
        self.model_intensity.to(self.device)
        
        self.model_severity.eval()
        self.model_intensity.eval()
    
    def predict(self, earthquake_magnitude, latitude, longitude, water_height=None, 
                tsunami_magnitude_abe=None, tsunami_magnitude_iida=None, intensity=None, 
                year=2024, month=1, day=1, hour=0, minute=0, second=0, 
                event_validity=4, cause_code=1, vol=0, num_runups=1):
        
        input_data = [year, month, day, hour, minute, second, event_validity, cause_code, 
                      earthquake_magnitude, vol, latitude, longitude]
        input_data.append(water_height if water_height is not None else 0)
        input_data.append(num_runups)
        input_data.append(tsunami_magnitude_abe if tsunami_magnitude_abe is not None else 0)
        input_data.append(tsunami_magnitude_iida if tsunami_magnitude_iida is not None else 0)
        input_data.append(intensity if intensity is not None else 0)
        
        while len(input_data) < self.input_size:
            input_data.append(0)
        input_data = input_data[:self.input_size]
        
        input_tensor = torch.FloatTensor(
            self.scaler.transform(self.imputer.transform(np.array(input_data).reshape(1, -1)))
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
