import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model():
    # save dataset 
    df = pd.read_csv("data/student_data.csv")
    
    x = df.drop(["pass"],axis=1)
    y = df["pass"]
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=44)
    LR = LogisticRegression()
    LR.fit(x_train,y_train)
    
    y_hat = LR.predict(x_test)
    acc = accuracy_score(y_test,y_hat)
    print("Model accuracy = ",acc)
    
    # create model folder
    os.makedirs("models",exist_ok=True)
    
    # save model
    model_path = "models\model.pkl"
    joblib.dump(LR, model_path)
    
    print("Model trained and save successfully")
    
if __name__=="__main__":
    train_model()