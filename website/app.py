from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


with open("model.bin", 'rb') as f_in:
    model = pickle.load(f_in)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        age = int(request.form['age'])
        sex = str(request.form['sex'])
        chestpaintype = str(request.form['chestpaintype'])
        restingbp = int(request.form['restingbp'])
        cholesterol = int(request.form['cholesterol'])
        fastingbs = int(request.form['fastingbs'])
        restingecg = str(request.form['restingecg'])
        maxhr = int(request.form['maxhr'])
        exerciseangina = int(request.form['exerciseangina'])
        oldpeak = float(request.form['oldpeak'])
        stslope = str(request.form['stslope'])

        df = pd.DataFrame({
            "Age": [age],
            "Sex": [sex],
            "ChestPainType": [chestpaintype],
            "RestingBP": [restingbp],
            "Cholesterol": [cholesterol],
            "FastingBS": [fastingbs],
            "RestingECG": [restingecg],
            "MaxHR": [maxhr],
            "ExerciseAngina": [exerciseangina],
            "Oldpeak": [oldpeak],
            "ST_Slope": [stslope],
        })

        prediction = model.predict_proba(df)
        return render_template('predict.html', prediction=prediction[0][1], restingbp=restingbp,
                               cholesterol=cholesterol, fastingbs=fastingbs)


