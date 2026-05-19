import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # type: ignore
import pickle
import os

# Generate synthetic fraud detection dataset
def generate_synthetic_data(n_samples=5000):
    np.random.seed(42)
    
    # Create synthetic data
    data = {
        'amount': np.random.uniform(10, 10000, n_samples),
        'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer', 'deposit'], n_samples),
        'location': np.random.choice(['online', 'atm', 'branch', 'phone'], n_samples),
        'time': np.random.choice(['morning', 'afternoon', 'evening', 'night'], n_samples),
        'account_age': np.random.uniform(0.1, 30, n_samples),
    }
    
    # Generate target variable with some patterns
    df = pd.DataFrame(data)
    
    # Fraud is more likely with:
    # - High amounts, unusual times/locations, low account age
    fraud_conditions = (
        (df['amount'] > 5000) |
        (df['time'].isin(['night', 'evening'])) |
        (df['account_age'] < 1) |
        ((df['transaction_type'] == 'withdrawal') & (df['amount'] > 3000))
    )
    
    df['is_fraud'] = fraud_conditions.astype(int)
    
    # Add some random fraud to generalize
    random_fraud = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    df['is_fraud'] = np.logical_xor(df['is_fraud'].astype(bool), random_fraud.astype(bool)).astype(int)
    
    return df

# Preprocess data
def preprocess_data(df):
    df = df.copy()
    
    # Encode categorical variables
    le_type = LabelEncoder()
    le_location = LabelEncoder()
    le_time = LabelEncoder()
    
    df['transaction_type'] = le_type.fit_transform(df['transaction_type'])
    df['location'] = le_location.fit_transform(df['location'])
    df['time'] = le_time.fit_transform(df['time'])
    
    return df, {
        'transaction_type': le_type,
        'location': le_location,
        'time': le_time
    }

# Train model
def train_model():
    print("Generating synthetic dataset...")
    df = generate_synthetic_data(5000)
    
    print("Preprocessing data...")
    df_processed, encoders = preprocess_data(df)
    
    # Split features and target
    X = df_processed[['amount', 'transaction_type', 'location', 'time', 'account_age']]
    y = df_processed['is_fraud']
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save model
    with open('model/fraud_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save encoders
    with open('model/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    with open('model/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print("\nModel saved to model/fraud_model.pkl")
    print("Encoders saved to model/encoders.pkl")
    print("Metrics saved to model/metrics.pkl")

if __name__ == "__main__":
    train_model()
