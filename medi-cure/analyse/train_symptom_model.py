import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv('training_data.csv')

# The last column 'prognosis' is our target, the rest are symptoms (features)
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Encode the string labels (disease names) into numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
print("Training the model on the new dataset...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")

# Save the trained model, the encoder, and the list of symptoms
joblib.dump(model, 'symptom_disease_model.pkl')
joblib.dump(le, 'symptom_disease_encoder.pkl')
joblib.dump(list(X.columns), 'symptoms_list.pkl')

print("\nModel, encoder, and symptoms list have been saved to .pkl files.")