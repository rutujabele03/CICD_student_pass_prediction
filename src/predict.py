import joblib
import numpy as np
import os

def predict_model():
    
    model_path = "models\model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model does not found")
    
    model = joblib.load(model_path)
    
    sample_input = np.array([[4,80,9]])
    prediction = model.predict(sample_input)
    
    print("Student will pass(1) or fail(0) : ",prediction[0])
    return prediction

if __name__=="__main__":
    predict_model()
    