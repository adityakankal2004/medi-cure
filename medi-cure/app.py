from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the Flask application
app = Flask(__name__)

# --- 1. Load Models and Encoders ---

# Initialize all models and encoders to None
params_model, params_encoder, symptoms_model, symptoms_encoder = None, None, None, None
model_columns, symptom_columns = [], []

# Load Medical Parameters Model
try:
    params_model = joblib.load('medical_report_model.pkl')
    params_encoder = joblib.load('medical_report_encoder.pkl')
    model_columns = ['Age', 'Sex', 'BloodPressure', 'Cholesterol', 'MaxHeartRate', 'Glucose', 
                     'BMI', 'Albumin', 'Bilirubin', 'Alamine_ALT', 'Copper', 'Stage', 
                     'SpecificGravity', 'Hemoglobin']
    logging.info("✅ Medical Parameters model loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"❌ ERROR loading medical parameters model/encoder: {e}")


# Load Symptoms Model (REVISED FOR DIAGNOSTICS)
try:
    symptoms_model = joblib.load('svc.pkl')
    symptoms_encoder = joblib.load('le.pkl')
    
    # Load dataset to get symptom columns
    symptoms_df = pd.read_csv('dataset.csv')
    
    # --- Data Cleaning and Diagnostic Step ---
    # 1. Clean up column names to remove extra spaces or hidden characters
    symptoms_df.columns = symptoms_df.columns.str.strip()
    
    # 2. Create the list of symptoms, explicitly removing the disease column
    symptom_columns = [col for col in symptoms_df.columns if 'Disease' not in col and 'Unnamed' not in col]
    
    # 3. THIS IS THE IMPORTANT DIAGNOSTIC STEP: Print the list to the terminal
    print("--- SYMPTOMS LIST LOADED ---")
    print(symptom_columns)
    print(f"Total symptoms found: {len(symptom_columns)}")
    print("----------------------------")
    
    logging.info(f"✅ Symptoms model loaded successfully with {len(symptom_columns)} symptoms.")

except FileNotFoundError as e:
    logging.error(f"❌ ERROR loading symptoms model/encoder: {e}")
except Exception as e:
    logging.error(f"❌ An unexpected error occurred while loading symptoms data: {e}")

# --- 2. Define Application Routes ---

@app.route('/')
def home():
    """Renders the main homepage with two choices."""
    return render_template('index.html')

@app.route('/parameters')
def parameters_page():
    """Renders the page for prediction using medical parameters."""
    return render_template('parameters.html')

@app.route('/symptoms')
def symptoms_page():
    """Renders the page for prediction using symptoms."""
    return render_template('symptoms.html')

@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    """Provides the list of symptoms to the frontend."""
    if not symptom_columns:
        return jsonify({'error': 'Symptoms list not available.'}), 500
    return jsonify(symptoms=symptom_columns)

# --- 3. Define Prediction API Endpoints ---

@app.route('/predict_parameters', methods=['POST'])
def predict_parameters():
    """API endpoint for the medical parameters model."""
    if not all([params_model, params_encoder]):
        return jsonify({'error': 'Medical parameters model is not loaded.'}), 500

    try:
        data = request.get_json(force=True)
        data_lower = {k.lower(): v for k, v in data.items()}
        
        input_values = []
        for col in model_columns:
            if col == 'Sex':
                sex_value = 1 if data_lower.get('sex') == 'Male' else 0
                input_values.append(sex_value)
            else:
                value = data_lower.get(col.lower())
                input_values.append(float(value) if value else np.nan)

        df = pd.DataFrame([input_values], columns=model_columns)
        
        # Simple imputation for any missing values
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
        
        prediction_encoded = params_model.predict(df)
        predicted_disease = params_encoder.inverse_transform(prediction_encoded)
        
        return jsonify({'predicted_disease': predicted_disease[0]})
    except Exception as e:
        logging.error(f"Error during parameter prediction: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_symptoms', methods=['POST'])
def predict_symptoms():
    """API endpoint for the symptoms model."""
    if not all([symptoms_model, symptoms_encoder]):
        return jsonify({'error': 'Symptoms model is not loaded.'}), 500

    try:
        data = request.get_json(force=True)
        user_symptoms = data.get('symptoms', [])
        
        # Create a feature vector (0s and 1s)
        test_data = pd.DataFrame(columns=symptom_columns)
        test_data.loc[0] = [1 if symptom in user_symptoms else 0 for symptom in symptom_columns]
        
        predict_disease = symptoms_model.predict(test_data)
        predicted_disease = symptoms_encoder.inverse_transform(predict_disease)
        
        return jsonify({'predicted_disease': predicted_disease[0]})
    except Exception as e:
        logging.error(f"Error during symptom prediction: {e}")
        return jsonify({'error': str(e)}), 400

# --- 4. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)