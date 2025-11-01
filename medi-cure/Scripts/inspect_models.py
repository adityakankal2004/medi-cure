import joblib
import os

paths = ['medical_report_model.pkl', 'analyse/symptom_disease_model.pkl']
for path in paths:
    print('---')
    if not os.path.exists(path):
        print(f"{path} : NOT FOUND")
        continue
    try:
        m = joblib.load(path)
        print(f"Loaded: {path}")
        print('Type:', type(m))
        # If sklearn Pipeline
        try:
            steps = getattr(m, 'steps', None)
            if steps:
                print('Pipeline steps:')
                for name, step in steps:
                    print(' -', name, ':', type(step).__name__)
            else:
                # Some objects may be a fitted model directly
                print('Estimator class:', type(m).__name__)
                # Check for common attributes
                if hasattr(m, 'estimators_'):
                    print('Has attribute estimators_')
        except Exception as e:
            print('Error inspecting pipeline steps:', e)
    except Exception as e:
        print(f"Error loading {path}: {e}")
print('Inspection complete')
