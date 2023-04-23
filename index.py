import streamlit as st
import time
import pandas as pd
import pickle
import numpy as np

choice=st.sidebar.radio("Select Predictor: ",["Heart Disease Predictor","Diabetes Predictor"])


if choice=="Heart Disease Predictor": 

    st.header("Heart Disease Prediction")
    dt=pd.read_csv(r"HeartPrediction/heart_data.csv")
    if st.checkbox("Show Training Data"):st.dataframe(dt,height=200,width=800)

    ########################### TAKING MANUAL INPUTS ############################################
    s,a=st.columns([1,3])
    with s:sex=st.selectbox("Sex: ",("(1)Male","(0)Female"),index=1)
    if sex=="(1)Male": sex=1
    else: sex=0
    with a:age=st.slider("Age: ",min_value=20,max_value=100,step=1)
    
    c,f=st.columns(2)
    with c:cp=st.selectbox("Chest Pain Type(cp): ",("(0)Asymptomatic","(1)Atypical angina","(2)Non-anginal pain","(3)Typical angina"))
    if cp=="(0)Asymptomatic":cp=0
    elif cp=="(1)Atypical angina":cp=1
    elif cp=="(2)Non-anginal pain":cp=2
    else:cp=3
    with f:fbs=st.number_input("Fasting Blood Sugar(fbs): ",min_value=0.00,step=0.01)
    if fbs>120:fbs=1
    else:fbs=0

    trestbps=st.slider("Resting Blood pressure(trestbps): ",min_value=90,max_value=200,step=1)

    e,r=st.columns(2)
    with r:recg=st.selectbox("Resting ECG results(restecg): ",("(0)LV hypertrophy","(1)Normal","(2)ST-T wave abnormality"))
    if recg=="(0)LV hypertrophy":recg=0
    elif recg=="(1)Normal":recg=1
    else:recg=2
    with e:exang=st.radio("Exercise induced angina(exang): ",('(1)Yes','(0)No'))

    ch,th=st.columns(2)
    with ch:chol=st.number_input("Cholestrol(chol): ",min_value=0,step=1)
    with th:thalach=st.number_input("Maximum Heart rate(thalach): ",min_value=0,step=1)
    
    caa,o=st.columns(2)
    with caa:ca=st.selectbox("No of major vessels(ca): ",(0,1,2,3,4))
    if exang=='Yes':exang=1
    else:exang=0
    with o:oldpeak=st.number_input("Oldpeak: ",min_value=0.0,step=0.1)

    sl,tl=st.columns(2)
    with sl:slope=st.selectbox("Slope of the peak exercise ST segment(slope): ",('(0)Downsloping','(1)Flat','(2)Upsloping'))
    if slope=='(0)Downsloping':slope=0
    elif slope=='(1)Flat':slope=1
    else:slope=2
    with tl:thal=st.selectbox("Thalassemia Value(thal): ",("(0)NULL","(1)Fixed defect","(2)Normal blood flow","(3)Reversible defect"))
    if thal=="(0)NULL":thal=0
    elif thal=="(1)Fixed defect":thal=1
    elif thal=="(2)Normal blood flow":thal=2
    else:thal=3
    ################################ RUNNING THE LOADED MODELS ######################################
    filename1="trained_lr.sav"
    filename2="trained_svc.sav"
    filename3="trained_rfc.sav"
    loaded_lr=pickle.load(open(r"HeartPrediction/trained_lr.sav","rb"))
    loaded_svc=pickle.load(open(r"HeartPrediction/trained_svc.sav","rb"))
    loaded_rfc=pickle.load(open(r"HeartPrediction/trained_rfc.sav","rb"))

    loaded_models=[loaded_lr,loaded_svc,loaded_rfc]

    input_data=np.array([age,sex,cp,trestbps,chol,fbs,recg,thalach,exang,oldpeak,slope,ca,thal])
    input_data=input_data.reshape(1,-1)

    if st.button("Run Model"):
        counter=st.progress(0)
        for i in range(1,100):
            time.sleep(0.005)
            counter.progress(i+1)
        for i in loaded_models:
            st.write(str(i)+" predicts: ")
            if(i.predict(input_data)[0]==1): st.error("Heart Disease Detected")
            else: st.success("No heart disease detected")


else:
    st.header("Diabetes Prediction")
    df=pd.read_csv(r"DiabetesPrediction/diabetes_data.csv")
    if st.checkbox("Show Training Data"):st.dataframe(df,height=200,width=800)

    ########################### TAKING MANUAL INPUTS ############################################
    pr,a=st.columns([1,3])
    with pr:pregnancies=st.slider("Pregnancies: ",min_value=0,max_value=25,step=1)
    with a:age=st.slider("Age: ",min_value=20,max_value=100,step=1)
    #glucose in single column
    
    glucose=st.slider("Glucose: ",min_value=0,max_value=250,step=1)
    b,s=st.columns(2)
    with b:bp=st.slider("Blood Pressure: ",min_value=0,max_value=150,step=1)
    with s:skin=st.slider("Skin Thickness: ",min_value=0,max_value=100,step=1)

    i,bm,pe=st.columns(3)
    with i:insulin=st.number_input("Insulin: ",min_value=0,max_value=1000,step=1)
    with bm:bmi=st.number_input("BMI: ",min_value=0.0,max_value=100.0,step=0.1)
    with pe:pedigree=st.number_input("DiabetesPedigreeFunction: ",min_value=0.0,max_value=5.0,step=0.001)
    ################################ RUNNING THE LOADED MODELS ######################################
    filename1="trained_classifier.sav"
    filename2="trained_forest.sav"
    filename3="trained_logistic.sav"
    loaded_classifier=pickle.load(open(r"DiabetesPrediction/trained_classifier.sav","rb"))
    loaded_forest=pickle.load(open(r"DiabetesPrediction/trained_forest.sav","rb"))
    loaded_logistic=pickle.load(open(r"DiabetesPrediction/trained_logistic.sav","rb"))

    loaded_models=[loaded_classifier,loaded_forest,loaded_logistic]

    input_data=np.array([pregnancies,glucose,bp,skin,insulin,bmi,pedigree,age])
    input_data=input_data.reshape(1,-1)

    if st.button("Run Model"):
        counter=st.progress(0)
        for i in range(1,100):
            time.sleep(0.005)
            counter.progress(i+1)
        for i in loaded_models:
            st.write(str(i)+" predicts: ")
            if(i.predict(input_data)[0]==1): st.error("Diabetic")
            else: st.success("Non-Diabetic")
