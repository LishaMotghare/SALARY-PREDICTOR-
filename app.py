import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Load the trained model
model = pickle.load(open('best_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    years_experience = data['YearsExperience']

    # Create a DataFrame for prediction
    prediction_df = pd.DataFrame([years_experience], columns=['YearsExperience'])

    # Make prediction
    prediction = model.predict(prediction_df)

    return jsonify({'predicted_salary': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
