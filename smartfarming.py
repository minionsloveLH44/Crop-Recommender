import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("modified.csv")
a=data[["pH","Temperature","Rainfall"]]
b1=data["Soil Type"]
b2=data["Recommended crop"]
b3=data["Natural Fertilizer"]
train_a,test_a,train_b1,test_b1,train_b2,test_b2,train_b3,test_b3=train_test_split(a,b1,b2,b3,test_size=10/100,random_state=1)
scaler=StandardScaler()
trainsc=scaler.fit_transform(train_a)
testsc=scaler.transform(test_a)
md1=RandomForestClassifier(n_estimators=1000,random_state=1)
md1.fit(trainsc,train_b1)
md2=RandomForestClassifier(n_estimators=1000,random_state=1)
md2.fit(trainsc,train_b2)
md3=RandomForestClassifier(n_estimators=1000,random_state=1)
md3.fit(trainsc,train_b3)
st.title("🌱CROP RECOMMENDATION SYSTEM FOR SMART FARMING🌱")
st.write("This AI model recommends you the best crop that can be cultivated in your soil after analyzing the conditions below")
pH=st.number_input("Enter pH⚗️:", min_value=3.5, max_value=9.0, step=0.1)
Temperature=st.number_input("Enter temperature(degree C)🌞:", min_value=-50, max_value=50)
Rainfall=st.number_input("Enter Rainfall(mm)🌧 :",min_value=0, max_value=2000)
if st.button("Predict"):
    ipdata= pd.DataFrame([[pH,Temperature,Rainfall]], columns=["pH", "Temperature", "Rainfall"])
    scipdata=scaler.transform(ipdata)
    soil=md1.predict(scipdata)
    crop=md2.predict(scipdata)
    fert=md3.predict(scipdata)
    st.success(f"🌺Soil Type🌺:{soil[0]}")
    st.success(f"🌺🌾Recommended crop🌱🌺:{crop[0]}")
    st.success(f"🌿Natural Fertilizer🌿:{fert[0]}")
    st.write("🌷🌹💐Close the browser tab to exit the application🌸🌼🌻")
    st.write(f"### Why {crop[0]} fits Natural Farming 🌱")
    st.write(f"""
    - **Soil Compatibility**: This crop works well with the detected soil type ({soil[0]}).
    - **Water Efficiency**: Known for its water efficiency, making it suitable for natural water conservation practices.
    - **Organic Fertilizer Recommendations**: Compost and green manure can enhance growth without synthetic inputs.
    - **Biodiversity Support**: Encourages diversity in farming systems, beneficial for soil health and pest control.
    - **TIPS**:"Use clover, legumes, or vetch for natural soil protection 🌱.
               Rotate crops like tomatoes with beans to enrich soil naturally 🌿.
    """)
