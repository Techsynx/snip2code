# src/predict.py
import numpy as np
from src.model import create_model
from src.preprocess import preprocess_image

def load_model(model_path='models/trained_model.h5'):
    model = create_model()
    model.load_weights(model_path)
    return model

def predict_code(image_path, model):
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    
    # Decoding the prediction to code (simplified)
    code = ''.join([chr(int(np.argmax(pred))) for pred in prediction])
    return code

if __name__ == "__main__":
    model = load_model()
    image_path = "data/example_image.png"
    predicted_code = predict_code(image_path, model)
    print(f"Predicted HTML/CSS Code:\n{predicted_code}")
