📘 Human Activity Recognition (HAR) – ML Classification + Flask Web App

- This project builds a Human Activity Recognition (HAR) system using classical machine learning models and provides a Flask-based web interface for easy model prediction.

📌 Overview

- Problem Type: Multi-class classification
- Train Shape: 7352 × 17
- Test Shape: 2947 × 17
- Features: 16 numeric sensor readings
- Target: Activity Label
- Models Included: Logistic Regression, Random Forest, AdaBoost, Gradient Boosting, XGBoost
- Deployment Interface: Flask Web App


📂 Dataset

File	Description	Shape
- HAR_Train_.csv	Training dataset	(7352, 17)
- HAR_Test_.csv	Testing dataset	(2947, 17)

Both contain:

- 16 sensor signals (accelerometer + gyroscope)
- 1 target label: activity class

- Common activities include: Walking, Standing, Sitting, Laying, Upstairs / Downstairs

🧠 ML Workflow

1. Load Data

- train_df = pd.read_csv("HAR_Train_.csv")
- test_df = pd.read_csv("HAR_Test_.csv")


2. Split Features & Labels

- X_train = train_df.drop("Activity", axis=1)
- y_train = train_df["Activity"]

- X_test = test_df.drop("Activity", axis=1)
- y_test = test_df["Activity"]


3. Scaling

- scaler = StandardScaler()
- X_train_scaled = scaler.fit_transform(X_train)
- X_test_scaled = scaler.transform(X_test)


4. Train 5 ML Models

- Models stored using joblib.


📈 Model Evaluation

- Metrics used: Accuracy, Classification Report, Precision, Recall, F1-score, Confusion Matrix

- Ensemble models (Random Forest, XGBoost) usually perform best.


🌐 Flask Web Interface

- A simple Flask app is built to allow users to input sensor values and get the predicted activity.

App Features:

- User-friendly HTML form
- Inputs are passed to trained ML model
- Model returns predicted activity
- Clean UI with CSS


🚀 How to Run Locally

- Install Dependencies
pip install -r requirements.txt

- Run Flask App
python app.py

- Open in Browser
http://127.0.0.1:5000
