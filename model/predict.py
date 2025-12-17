import pickle
import pandas as pd 
import os

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path,'rb') as f:
    model = pickle.load(f)

class_labels = model.classes_.tolist()

#ml flow
ML_VERSION = '1.0.0'

def predict_output(user_input: dict):
    input_df = pd.DataFrame([user_input])
    
    # predict the class
    predicted_class = model.predict(input_df)[0]
    

    probabilities = model.predict_proba(input_df)[0]
    confidence = max(probabilities)
    class_probs = dict(zip(class_labels, map(lambda p:round(p,4), probabilities)))

    return {
        "predicted_category":predicted_class,
        "confidence":round(confidence,4),
        "class Probabilities": class_probs

    }
