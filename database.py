import json
import os
from datetime import datetime
from typing import List, Optional

# Simple in-memory database
users_db = {
    "user": {"username": "user", "password": "user123", "is_admin": False},
    "admin": {"username": "admin", "password": "admin123", "is_admin": True}
}

transactions_db = []
model_metrics = {}

# File paths
USERS_FILE = "data_users.json"
TRANSACTIONS_FILE = "data_transactions.json"
METRICS_FILE = "data_metrics.json"

def load_data():
    """Load data from files if they exist"""
    global users_db, transactions_db, model_metrics
    
    if os.path.exists(TRANSACTIONS_FILE):
        try:
            with open(TRANSACTIONS_FILE, 'r') as f:
                transactions_db = json.load(f)
        except:
            transactions_db = []
    
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                model_metrics = json.load(f)
        except:
            model_metrics = {}

def save_transactions():
    """Save transactions to file"""
    with open(TRANSACTIONS_FILE, 'w') as f:
        json.dump(transactions_db, f, indent=2)

def save_metrics():
    """Save metrics to file"""
    with open(METRICS_FILE, 'w') as f:
        json.dump(model_metrics, f, indent=2)

def add_transaction(user_id, amount, transaction_type, location, time, account_age, is_fraud, risk_score):
    """Add a new transaction"""
    transaction = {
        "id": len(transactions_db) + 1,
        "user_id": user_id,
        "amount": amount,
        "transaction_type": transaction_type,
        "location": location,
        "time": time,
        "account_age": account_age,
        "is_fraud": is_fraud,
        "risk_score": risk_score,
        "created_at": datetime.utcnow().isoformat()
    }
    transactions_db.append(transaction)
    save_transactions()
    return transaction

def get_all_transactions() -> List:
    """Get all transactions"""
    return transactions_db

def get_transaction(transaction_id):
    """Get a single transaction"""
    for t in transactions_db:
        if t["id"] == transaction_id:
            return t
    return None

def delete_transaction(transaction_id) -> bool:
    """Delete a transaction"""
    global transactions_db
    transactions_db = [t for t in transactions_db if t["id"] != transaction_id]
    save_transactions()
    return True

def get_user(username: str):
    """Get user by username"""
    return users_db.get(username)

def set_metrics(accuracy, precision, recall, f1_score):
    """Set model metrics"""
    global model_metrics
    model_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "trained_at": datetime.utcnow().isoformat()
    }
    save_metrics()

def get_metrics():
    """Get model metrics"""
    return model_metrics

# Load initial data
load_data()
