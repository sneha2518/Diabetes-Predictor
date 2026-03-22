from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        input_data = np.array(features).reshape(1, -1)

        prediction = model.predict(input_data)[0]

        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return render_template("index.html", prediction_text=f"Result: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error in input: {str(e)} ")

if __name__ == "__main__":
    app.run(debug=True)

    