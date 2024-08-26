# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from flask import Flask,request,render_template
import pickle
app=Flask(__name__)
#loadthe pickle model
model=pickle.load(open("model.pkl",'rb'))

@app.route("/")
def home():
    return render_template("indexx.html")
@app.route('/predict',methods=['POST'])
def predict():
    
    inputdata=[float(x) for x in request.form.values()]
    features=[np.array(inputdata)]
    #input_data_as_numpy_array= np.asarray(inputdata)
    #input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = model.predict(features)
    if (prediction[0] == 1):
        prediction="***you are placed***"
    else:
        prediction="***you are not placed***"
   
    return render_template('indexx.html',output= "result:"+prediction)
if __name__ == "__main__":
    app.run(debug=True) 

#python deploy.py
#python temp.py
#http://127.0.0.1:5000/

