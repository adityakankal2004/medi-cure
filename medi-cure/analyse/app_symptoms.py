from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the saved model, encoder, and symptoms list
model = joblib.load('symptom_disease_model.pkl')
encoder = joblib.load('symptom_disease_encoder.pkl')
symptoms = joblib.load('symptoms_list.pkl')

@app.route('/')
def home():
    # Pass the list of symptoms to the HTML template to create the checklist
    return render_template('index_symptoms.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the list of selected symptoms from the form
        selected_symptoms = request.get_json()['symptoms']
        
        # Create a binary input vector (a row of 0s and 1s) based on the selected symptoms
        input_vector = np.zeros(len(symptoms))
        for symptom in selected_symptoms:
            if symptom in symptoms:
                index = symptoms.index(symptom)
                input_vector[index] = 1
        
        # Reshape for the model and create a DataFrame
        input_df = pd.DataFrame([input_vector], columns=symptoms)
        
        # Make prediction
        prediction_encoded = model.predict(input_df)
        
        # Decode the prediction back to the disease name
        predicted_disease = encoder.inverse_transform(prediction_encoded)
        
        return jsonify({'predicted_disease': predicted_disease[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)