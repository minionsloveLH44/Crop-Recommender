import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("soil_rec.csv")
a=data[["pH","Temperature","Rainfall"]]
b1=data["Soil Type"]
b2=data["Recommended crop"]
train_a,test_a,train_b1,test_b1,train_b2,test_b2=train_test_split(a,b1,b2,test_size=10/100,random_state=1)
scaler=StandardScaler()
trainsc=scaler.fit_transform(train_a)
testsc=scaler.transform(test_a)
md1=RandomForestClassifier(n_estimators=1000,random_state=1)
md1.fit(trainsc,train_b1)
md2=RandomForestClassifier(n_estimators=1000,random_state=1)
md2.fit(trainsc,train_b2)
st.title("🌱CROP RECOMMENDATION SYSTEM FOR SMART FARMING🌱")
st.write("This AI model recommends you the best crop that can be cultivated in your soil after analyzing the conditions below")
pH=st.number_input("Enter pH⚗️:", min_value=3.0, max_value=9.5, step=0.1)
Temperature=st.number_input("Enter temperature(degree C)🌞:", min_value=-50, max_value=50)
Rainfall=st.number_input("Enter Rainfall(mm)🌧 :",min_value=0, max_value=2000)
if st.button("Predict"):
    ipdata= pd.DataFrame([[pH,Temperature,Rainfall]], columns=["pH", "Temperature", "Rainfall"])
    scipdata=scaler.transform(ipdata)
    soil=md1.predict(scipdata)
    crop=md2.predict(scipdata)
    st.success(f"🌺Soil Type🌺:{soil[0]}")
    st.success(f"🌾Recommended crop🌾:{crop[0]}")
    st.write("🌷🌹💐Close the browser tab to exit the application🌸🌼🌻")
