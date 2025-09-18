import pandas as pd

# Define the name of the dataset file
filename = 'training_data.csv'

try:
    # Load the dataset
    df = pd.read_csv(filename)
    print(f"Dataset '{filename}' loaded successfully.")

    # The target column with the disease names is 'prognosis'
    target_column = 'prognosis'

    # Check if the prognosis column exists
    if target_column in df.columns:
        # Count the number of unique diseases
        disease_count = df[target_column].nunique()
        
        # Get the list of all unique disease names
        disease_names = df[target_column].unique()
        
        print("\n--------------------------------------------------")
        print(f"This dataset can be used to predict {disease_count} different diseases.")
        print("--------------------------------------------------")
        
        print("\nHere is the complete list of diseases:")
        # Print the list in a clean, sorted format
        for name in sorted(disease_names):
            print(f"- {name}")
    else:
        print(f"Error: The expected target column '{target_column}' was not found in the dataset.")
        print(f"Available columns are: {list(df.columns)}")

except FileNotFoundError:
    print(f"Error: '{filename}' not found.")
    print("Please make sure this script is in the same folder as your dataset.")
except Exception as e:
    print(f"An error occurred: {e}")