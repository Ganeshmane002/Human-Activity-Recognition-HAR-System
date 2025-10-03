from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("logistic_model_baseline.joblib")
scaler = joblib.load("scaler_baseline.joblib")

columns = [
    'tGravityAcc-mean()-X', 'tGravityAcc-min()-X', 'angle(X,gravityMean)',
    'tGravityAcc-mean()-Y', 'angle(Y,gravityMean)', 'tGravityAcc-max()-Y',
    'tGravityAcc-min()-Y', 'tGravityAcc-energy()-X', 'tGravityAcc-max()-X',
    'tGravityAcc-energy()-Y', 'tGravityAccMag-std()',
    'fBodyAccJerk-bandsEnergy()-1,16', 'tGravityAcc-arCoeff()-Z,2',
    'tGravityAcc-min()-Z', 'tGravityAcc-mean()-Z'
]

def get_default_values():
    return {
        'tGravityAcc-mean()-X': 0.279653,
        'tGravityAcc-min()-X': -0.998245,
        'angle(X,gravityMean)': 0.034469,
        'tGravityAcc-mean()-Y': 0.062119,
        'angle(Y,gravityMean)': 0.059256,
        'tGravityAcc-max()-Y': 0.986765,
        'tGravityAcc-min()-Y': -0.984542,
        'tGravityAcc-energy()-X': 0.057288,
        'tGravityAcc-max()-X': 0.976324,
        'tGravityAcc-energy()-Y': 0.047891,
        'tGravityAccMag-std()': 0.106556,
        'fBodyAccJerk-bandsEnergy()-1,16': 0.324322,
        'tGravityAcc-arCoeff()-Z,2': 0.228912,
        'tGravityAcc-min()-Z': -0.936543,
        'tGravityAcc-mean()-Z': 0.102543
    }

activity_labels = {
    1: "The Person is WALKING",
    2: "The Person is WALKING_UPSTAIRS",
    3: "The Person is WALKING_DOWNSTAIRS",
    4: "The Person is SITTING",
    5: "The Person is STANDING",
    6: "The Person is LYING"
}

@app.route("/")
def home():
    return render_template("home.html", title="Home")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    default_values = get_default_values()
    if request.method == "POST":
        input_data = [float(request.form[col]) for col in columns]
        input_df = pd.DataFrame([input_data], columns=columns)
        input_scaled = scaler.transform(input_df)
        pred_class = model.predict(input_scaled)[0]
        prediction = activity_labels.get(pred_class, "Unknown")
    return render_template("predict.html", title="Predict", columns=columns, default_values=default_values, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)