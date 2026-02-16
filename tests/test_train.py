import os
import joblib
from src.train import train_model

def test_model_training():
    
    train_model()
    
    model_path = "models\model.pkl"
    assert os.path.exists(model_path), "Model File Not found"
    
    model = joblib.load(model_path)
    
    assert hasattr(model, "predict"), "Model does not have predict method"
    accuracy = train_model()
    assert accuracy>=0.80, "Accuracy below 0.8! Pipeline fails"