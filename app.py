from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle


app=Flask(__name__)#app initialization

@app.route('/')
def index():
    return render_template("home.html")#for home page

@app.route("/result",methods=['POST','GET'])

def result():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    #scale down the data
    x = np.array([gender,age,hypertension,heart_disease,ever_married,work_type,residence_type,
                  avg_glucose_level,bmi,smoking_status]).reshape(1, -1) #reshape the data in 2d array

    scaler_path=os.path.join('C:/Users/Admin/Desktop/git/heart-stroke-detection','model/scalar.pkl')
    scaler=None
    # read binary
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)#scaling down the input data

    #same as we did in jupyter
    x=scaler.transform(x)

    model_path=os.path.join('C:/Users/Admin/Desktop/git/heart-stroke-detection','model/dt.sav')
    dt=joblib.load(model_path)

    y_pred=dt.predict(x)

    if y_pred==0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')

if __name__=="__main__":
    app.run(debug=True,port=5000)



