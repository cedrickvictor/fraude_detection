from fastapi import FastAPI, HTTPException, Request, Form  # type: ignore
from fastapi.responses import HTMLResponse, RedirectResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore
from datetime import datetime
import pickle
import os
import database as db

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Favicon handler to remove 404 errors
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return RedirectResponse(url="/static/favicon.ico", status_code=301)

# Global variables
model = None
encoders = None
metrics = None
current_user = None

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model, encoders, metrics
    
    if os.path.exists('model/fraud_model.pkl'):
        with open('model/fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('model/metrics.pkl', 'rb') as f:
            metr = pickle.load(f)
            db.set_metrics(metr['accuracy'], metr['precision'], metr['recall'], metr['f1_score'])
        print("✓ Model loaded successfully!")
    else:
        print("✗ Model file not found. Please run train.py first.")

# ==================== AUTHENTICATION ====================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    global current_user
    
    # Check credentials
    user = db.get_user(username)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if user.get("password") == password:
        current_user = username
        if user.get("is_admin"):
            return RedirectResponse(url="/admin", status_code=303)
        else:
            return RedirectResponse(url="/dashboard", status_code=303)
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/logout")
async def logout():
    global current_user
    current_user = None
    return RedirectResponse(url="/", status_code=303)

# ==================== DASHBOARD ====================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    
    transactions = db.get_all_transactions()
    total_transactions = len(transactions)
    fraud_count = len([t for t in transactions if t.get("is_fraud")])
    safe_count = total_transactions - fraud_count
    
    metrics_data = db.get_metrics()
    accuracy = metrics_data.get('accuracy', 0) * 100 if metrics_data else 0
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "username": current_user,
        "total_transactions": total_transactions,
        "fraud_count": fraud_count,
        "safe_count": safe_count,
        "accuracy": f"{accuracy:.2f}",
        "transactions": transactions
    })

# ==================== TRANSACTION PREDICTION ====================

@app.get("/transaction", response_class=HTMLResponse)
async def transaction_page(request: Request):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("transaction.html", {"request": request, "username": current_user})

@app.post("/predict")
async def predict(
    amount: float = Form(...),
    transaction_type: str = Form(...),
    location: str = Form(...),
    time: str = Form(...),
    account_age: float = Form(...)
):
    if model is None or encoders is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Encode input
    try:
        encoded_type = encoders['transaction_type'].transform([transaction_type])[0]
        encoded_location = encoders['location'].transform([location])[0]
        encoded_time = encoders['time'].transform([time])[0]
    except:
        return {
            "error": "Invalid input value",
            "is_fraud": False,
            "risk_score": "0.00%"
        }
    
    # Prepare features
    features = [[amount, encoded_type, encoded_location, encoded_time, account_age]]
    
    # Predict
    prediction = model.predict(features)[0]
    risk_score = model.predict_proba(features)[0][1]
    
    # Save to database
    db.add_transaction(
        user_id=1,
        amount=amount,
        transaction_type=transaction_type,
        location=location,
        time=time,
        account_age=account_age,
        is_fraud=bool(prediction),
        risk_score=risk_score
    )
    
    return {
        "is_fraud": bool(prediction),
        "risk_score": f"{risk_score * 100:.2f}%",
        "message": "⚠️ Fraudulent Transaction Detected!" if prediction == 1 else "✓ Safe Transaction"
    }

# ==================== ADMIN PANEL ====================

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    if not current_user or not db.get_user(current_user).get("is_admin"):
        return RedirectResponse(url="/login", status_code=303)
    
    transactions = db.get_all_transactions()
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "transactions": transactions,
        "username": current_user
    })

@app.get("/admin/fraud-only")
async def fraud_only():
    transactions = [t for t in db.get_all_transactions() if t.get("is_fraud")]
    return {"transactions": transactions}

@app.delete("/transaction/{transaction_id}")
async def delete_transaction(transaction_id: int):
    if db.delete_transaction(transaction_id):
        return {"message": "Transaction deleted"}
    raise HTTPException(status_code=404, detail="Transaction not found")

@app.get("/export-data")
async def export_data():
    transactions = db.get_all_transactions()
    data = [{
        "id": t.get("id"),
        "amount": t.get("amount"),
        "type": t.get("transaction_type"),
        "location": t.get("location"),
        "time": t.get("time"),
        "account_age": t.get("account_age"),
        "is_fraud": t.get("is_fraud"),
        "risk_score": t.get("risk_score")
    } for t in transactions]
    return {"data": data}

# ==================== TRAIN MODEL ====================

@app.post("/train-model")
async def train_model_endpoint():
    import subprocess
    result = subprocess.run(["/usr/local/bin/python3", "train.py"], capture_output=True, text=True)
    
    # Reload model
    global model, encoders
    if os.path.exists('model/fraud_model.pkl'):
        with open('model/fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('model/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('model/metrics.pkl', 'rb') as f:
            metr = pickle.load(f)
            db.set_metrics(metr['accuracy'], metr['precision'], metr['recall'], metr['f1_score'])
    
    return {"message": "Model trained successfully", "output": result.stdout}

# ==================== API ENDPOINTS ====================

@app.get("/api/transactions")
async def get_transactions():
    return db.get_all_transactions()

@app.get("/api/stats")
async def get_stats():
    transactions = db.get_all_transactions()
    total = len(transactions)
    fraud_count = len([t for t in transactions if t.get("is_fraud")])
    safe_count = total - fraud_count
    
    metrics_data = db.get_metrics()
    
    return {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "safe_count": safe_count,
        "accuracy": metrics_data.get('accuracy', 0) * 100 if metrics_data else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
