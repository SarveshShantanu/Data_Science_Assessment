import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to preprocess claim descriptions
def preprocess_claims(claims):
    
    vectorizer = TfidfVectorizer(stop_words='english')
    claims_tfidf = vectorizer.fit_transform(claims)
    return claims_tfidf

# Loading the trained models
def load_models():
    # Update these paths with the correct location where your models are saved
    try:
        model_cov = joblib.load(r"best_rf_accident_model.pkl")  # path for Coverage Code model
        model_acc = joblib.load(r"best_rf_accident_model.pkl")  # path for Accident Source model
        return model_cov, model_acc
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while loading the models: {e}")
        return None, None

# Function to run the prediction
def run_model():
    try:
        # Open file dialog to select the input file
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return  # If no file is selected, return

        # Load the dataset
        df = pd.read_excel(file_path)
        
        if 'Claim Description' not in df.columns:
            messagebox.showerror("Error", "The dataset must contain a 'Claim Description' column.")
            return
        
        # Preprocess the claim descriptions (you can customize this function as per your needs)
        claims = df['Claim Description'].fillna('')  # Handling missing values in claim descriptions
        claims_tfidf = preprocess_claims(claims)
        
        # Load models
        model_cov, model_acc = load_models()
        if model_cov is None or model_acc is None:
            return  # Exit if models couldn't be loaded

        # Make predictions using the models
        cov_predictions = model_cov.predict(claims_tfidf)
        accident_predictions = model_acc.predict(claims_tfidf)

        # Add predictions as new columns in the dataframe
        df['Coverage Code Prediction'] = cov_predictions
        df['Accident Source Prediction'] = accident_predictions
        
        # Save the results to a new Excel file
        output_file = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if output_file:
            df.to_excel(output_file, index=False)
            messagebox.showinfo("Success", f"Predictions saved to {output_file}")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create the main window
root = tk.Tk()
root.title("Coverage Code and Accident Source Prediction")

# Set window size
root.geometry("400x200")

# Create and pack the run button
run_button = tk.Button(root, text="Run Model", width=20, command=run_model)
run_button.pack(pady=50)

# Start the Tkinter event loop
root.mainloop()
