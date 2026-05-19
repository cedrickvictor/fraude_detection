# 🏦 BK Bank Fraud Detection System

This project is a web based fraud detection system for online banking using Machine Learning.
The system try to detect if a banking transaction is fraud or not using RandomForest model.
It was developed as student project to simulate how banks like BK Bank can use ML to improve security.

## Features

* Real time fraud detection when user enter transaction
* Login system with normal user and admin
* Dashboard with statistics and charts
* Transaction form to test fraud prediction
* Admin panel to see all transactions
* Model accuracy, precision, recall display
* Export transaction data
* Retrain model from dashboard

## Technologies used

Backend: FastAPI (Python)
Frontend: HTML, CSS, Bootstrap, JavaScript
Machine Learning: scikit-learn, pandas, numpy
Database: JSON file storage
Server: Uvicorn

## Requirements

* Python 3.10 or higher
* pip installed

## Installation steps

Go to project folder

```bash
cd ~/Desktop/"new ML"/fraud_detection_system
```

Create virtual env (optional)

```bash
python3 -m venv venv
source venv/bin/activate
```

Install packages

```bash
pip install -r requirements.txt
```

Train the model first

```bash
python3 train.py
```

This will create files inside model folder

* fraud_model.pkl
* encoders.pkl
* metrics.pkl

Run server

```bash
python3 -m uvicorn main:app --port 9000 --reload
```

Open in browser

http://localhost:9000

## Demo login

User
username: user
password: user123

Admin
username: admin
password: admin123

## Project structure

fraud_detection_system/

main.py → FastAPI routes
database.py → fake database using json
train.py → train ML model
templates → html pages
static → css files
model → saved ML model
data → transactions data

## Pages

index.html → home page
login.html → login page
dashboard.html → dashboard
transaction.html → enter transaction
admin.html → admin page

## How system works

1. User login
2. User enter transaction
3. Data sent to backend
4. Model predict fraud or not
5. Risk score calculated
6. Saved in database
7. Show result on screen

## Features used for prediction

amount
transaction type
location
time of day
account age

## Model result

Accuracy around 94%
Precision around 94%
Recall around 98%
F1 score around 96%

Results may change because dataset is generated random.

## Notes

This project is for learning purpose only
Security is not perfect
Password stored in plain text
Not for real bank use

## Problems I faced

* Model not loading sometimes
* Port already used error
* Json file not saving
* Template not showing

Solved by reinstalling packages and retraining model

## Run again quick

```bash
pip install -r requirements.txt
python3 train.py
python3 -m uvicorn main:app --port 9000 --reload
```

## Author

Student project
Machine Learning + Web development practice


