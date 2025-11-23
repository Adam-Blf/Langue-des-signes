import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import os

# Define paths
DATA_PATH = 'machine_learning/data.csv' # Fixed path relative to root if run from root
MODEL_PATH = 'machine_learning/model.p'

def train():
    # Handle paths whether run from root or machine_learning dir
    data_path = DATA_PATH if os.path.exists(DATA_PATH) else 'data.csv'
    model_path = MODEL_PATH if os.path.dirname(MODEL_PATH) else 'model.p'

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please collect data first.")
        return

    print(f"Loading data from {data_path}...")
    try:
        # Read without header, as app.py appends rows directly
        df = pd.read_csv(data_path, header=None)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.empty:
        print("Dataset is empty.")
        return

    # Separate features and labels
    # Format: [Label, Feature1, Feature2, ..., Feature63]
    y = df.iloc[:, 0].values      # First column is Label
    X = df.iloc[:, 1:].values     # Rest are Features

    print(f"Data shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

    # Define models to train
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True), # probability=True needed for predict_proba
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print(f"  -> Accuracy: {score * 100:.2f}%")
            
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        except Exception as e:
            print(f"  -> Failed to train {name}: {e}")

    if best_model:
        print(f"\n🏆 Best Model: {best_name} with {best_score * 100:.2f}% accuracy")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump({'model': best_model, 'type': best_name, 'accuracy': best_score}, f)
        
        print(f"Model saved to {model_path}")
    else:
        print("Training failed for all models.")

if __name__ == "__main__":
    train()
