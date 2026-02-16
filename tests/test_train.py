import os
import joblib
from src.train import train_model,acc

def test_model_training():
    
    train_model()
    
    model_path = "models\model.pkl"
    assert os.path.exists(model_path), "Model File Not found"
    
    model = joblib.load(model_path)
    
    assert hasattr(model, "predict"), "Model does not have predict method"
    assert acc > 0.80, "Accuracy is less than 80! Failing pipeline"