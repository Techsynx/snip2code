from flask import Flask, request, render_template, url_for
from src.predict import load_model, predict_code

app = Flask(__name__)
model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    file.save(f"data/{file.filename}")
    predicted_code = predict_code(f"data/{file.filename}", model)
    return f"<pre>{predicted_code}</pre>"

if __name__ == "__main__":
    app.run(debug=True)
