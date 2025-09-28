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
    symptoms_model = joblib.load('analyse/symptom_disease_model.pkl')
    symptoms_encoder = joblib.load('analyse/symptom_disease_encoder.pkl')
    symptom_columns = joblib.load('analyse/symptoms_list.pkl')
    
    # --- Data Cleaning and Diagnostic Step ---
    # 1. Clean up column names to remove extra spaces or hidden characters
    symptom_columns = [col.strip() for col in symptom_columns]
    
    # 2. THIS IS THE IMPORTANT DIAGNOSTIC STEP: Print the list to the terminal
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
    
    # Clean and format symptoms for better display
    formatted_symptoms = []
    for symptom in symptom_columns:
        # Convert underscores to spaces and capitalize words
        formatted_name = symptom.replace('_', ' ').title()
        # Handle special cases
        formatted_name = formatted_name.replace('Alt', 'ALT')
        formatted_name = formatted_name.replace('Bmi', 'BMI')
        formatted_name = formatted_name.replace('Url', 'URL')
        
        formatted_symptoms.append({
            'value': symptom,
            'display': formatted_name
        })
    
    # Sort symptoms alphabetically by display name
    formatted_symptoms.sort(key=lambda x: x['display'])
    
    return jsonify(symptoms=formatted_symptoms)

# --- 3. Define Prediction API Endpoints ---

@app.route('/predict_parameters', methods=['POST'])
def predict_parameters():
    """API endpoint for the medical parameters model."""
    if not all([params_model, params_encoder]):
        return jsonify({'error': 'Medical parameters model is not loaded.'}), 500

    try:
        data = request.get_json(force=True)
        data_lower = {k.lower(): v for k, v in data.items()}
        
        # Debug logging
        logging.info(f"Received data: {data}")
        logging.info(f"Processed data: {data_lower}")
        
        # Convert sex to lowercase for validation
        if 'sex' in data_lower and data_lower['sex']:
            data_lower['sex'] = data_lower['sex'].lower()
        
        # Validation ranges for medical parameters
        validation_ranges = {
            'age': (0, 120),
            'bloodpressure': (50, 250),
            'cholesterol': (100, 500),
            'maxheartrate': (40, 220),
            'glucose': (50, 500),
            'bmi': (10, 80),
            'albumin': (1.0, 6.0),
            'bilirubin': (0.1, 20.0),
            'alamine_alt': (5, 500),
            'copper': (10, 200),
            'stage': (1, 5),
            'specificgravity': (1.000, 1.040),
            'hemoglobin': (5, 25)
        }
        
        # Validate input data
        validation_errors = []
        
        # Check if sex is provided (required)
        sex_value = data_lower.get('sex')
        if not sex_value or sex_value not in ['male', 'female']:
            validation_errors.append("Sex is required and must be 'Male' or 'Female'")
        
        # Check if age is provided (required)
        age_value = data_lower.get('age')
        if not age_value or age_value == '' or age_value == 'None':
            validation_errors.append("Age is required")
        else:
            try:
                age_num = float(age_value)
                if age_num < 0 or age_num > 120:
                    validation_errors.append("Age must be between 0 and 120")
            except (ValueError, TypeError):
                validation_errors.append("Age must be a valid number")
        
        # Count how many additional parameters are provided
        additional_params_count = 0
        for col in model_columns:
            if col == 'Sex' or col == 'Age':
                continue
            
            # Handle case sensitivity for Alamine_ALT
            col_key = col.lower()
            if col == 'Alamine_ALT':
                col_key = 'alamine_alt'
            
            value = data_lower.get(col_key)
            if value is not None and value != '' and value != 'None':
                additional_params_count += 1
        
        # Require at least 2 additional parameters (besides sex and age)
        if additional_params_count < 2:
            validation_errors.append("Please provide at least 2 additional medical parameters for accurate prediction")
        
        # Validate numeric parameters (only if provided)
        for col in model_columns:
            if col == 'Sex':
                continue
                
            # Handle case sensitivity for Alamine_ALT
            col_key = col.lower()
            if col == 'Alamine_ALT':
                col_key = 'alamine_alt'
                
            value = data_lower.get(col_key)
            if value is None or value == '' or value == 'None':
                continue  # Skip validation for empty fields
                
            try:
                num_value = float(value)
                min_val, max_val = validation_ranges[col_key]
                
                if num_value < min_val or num_value > max_val:
                    validation_errors.append(f"{col} must be between {min_val} and {max_val}")
                    
            except (ValueError, TypeError):
                validation_errors.append(f"{col} must be a valid number")
        
        if validation_errors:
            logging.info(f"Validation errors: {validation_errors}")
            return jsonify({'error': 'Validation failed', 'details': validation_errors}), 400
        
        # Process validated data with imputation for missing values
        input_values = []
        for col in model_columns:
            if col == 'Sex':
                # Default to 0 (Female) if not provided
                sex_value = 1 if data_lower.get('sex') == 'male' else 0
                input_values.append(sex_value)
            else:
                # Handle case sensitivity for Alamine_ALT
                col_key = col.lower()
                if col == 'Alamine_ALT':
                    col_key = 'alamine_alt'
                    
                value = data_lower.get(col_key)
                if value is None or value == '' or value == 'None':
                    input_values.append(np.nan)  # Use NaN for missing values
                else:
                    input_values.append(float(value))

        df = pd.DataFrame([input_values], columns=model_columns)
        
        # Impute missing values with mean for numeric columns
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
        
        # Make prediction
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