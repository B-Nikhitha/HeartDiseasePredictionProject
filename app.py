from flask import Flask,render_template,request
import pickle
import numpy as np

model=pickle.load(open('iri.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    result=''
    return render_template('templates/index.html',**locals())

@app.route('/predict',methods=['POST','GET'])
def predict():
    d1=float(request.form['f1'])
    d2=float(request.form['f2'])
    d3=float(request.form['f3'])
    d4=float(request.form['f4'])
    d5=float(request.form['f5'])
    d6=float(request.form['f6'])
    d7=float(request.form['f7'])
    d8=float(request.form['f8'])
    d9=float(request.form['f9'])
    d10=float(request.form['f10'])
    d11=float(request.form['f11'])
    d12=float(request.form['f12'])
    d13=float(request.form['f13'])
    d14=float(request.form['f14'])
    d15=float(request.form['f15'])
    y=model.predict([[d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15]])[0]
    if y==1:
        result='Yes'
    else:
        result='NO'
    return render_template('templates/index.html',**locals())

if __name__=="__main__":
    app.run(debug=True)
