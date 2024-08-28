import pandas as pd
import pickle
dt=pd.read_csv('C:/Users/badam/OneDrive/Desktop/HeartDiseasePrediction/heart_disease_dataset.csv')
print(dt)
dt['Gender']=dt['Gender'].map({'Male':1,'Female':0})
dt['Smoking']=dt['Smoking'].map({'Current':2,'Former':1,'Never':0})
dt['Alcohol Intake']=dt['Alcohol Intake'].map({'Heavy':2,'Moderate':1,'None':0})
dt['Family History']=dt['Family History'].map({'Yes':1,'No':0})
dt['Diabetes']=dt['Diabetes'].map({'Yes':1,'No':0})
dt['Obesity']=dt['Obesity'].map({'Yes':1,'No':0})
dt['Exercise Induced Angina']=dt['Exercise Induced Angina'].map({'Yes':1,'No':0})
dt['Chest Pain Type']=dt['Chest Pain Type'].map({'Typical Angina':0,'Atypical Angina':1,'Non-anginal Pain':2,'Asymptomatic':3})
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
X=dt.drop(['Heart Disease'],axis=1)
y=dt['Heart Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
pickle.dump(model,open('iri.pkl','wb'))
