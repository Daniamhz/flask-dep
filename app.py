import os
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Ensure paths are correct
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

# Load the trained model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from the form
            data = [float(request.form[f'feature{i}']) for i in range(1, 10)]

            # Ensure that the correct number of features are provided
            if len(data) != 9:
                return render_template('index.html', prediction_text='Error: The model expects 9 features.')
            
            # Transform the data and make prediction
            final_input = scaler.transform([np.array(data)])
            prediction = model.predict(final_input)
            
            # Return prediction to the user
            return render_template('index.html', prediction_text=f'Prediction: {prediction[0]}')
        
        except Exception as e:
            # Handling exceptions and displaying error messages
            return render_template('index.html', prediction_text=f'Error: {str(e)}')
    
    return render_template('index.html', prediction_text='Error: Only POST requests are allowed.')

if __name__ == "__main__":
    app.run(debug=True, port=5002)
