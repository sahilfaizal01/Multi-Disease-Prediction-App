import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
from keras.layers import *
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
     
def transformVal(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    std_data = scaler.fit_transform(input_data_reshaped)
    return std_data

diabetes_model = pickle.load(open('diabetes_svm_model.sav','rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav','rb'))
parkinson_disease_model = pickle.load(open('parkinson_model.sav','rb'))
#breast cancer
tf.random.set_seed(3)
model = tf.keras.Sequential([
                          tf.keras.layers.Flatten(input_shape=(30,)),
                          tf.keras.layers.Dense(20, activation='relu'),
                          tf.keras.layers.Dense(20, activation='relu'),
                          tf.keras.layers.Dense(2, activation='sigmoid')
])
model.load_weights('Breast_Cancer_model.h5')

with st.sidebar:
    selected = option_menu('Multi-Disease Prediction System',
                            ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinson Disease Prediction',
                            'Breast Cancer Detection'],
                            default_index=0)

if(selected == 'Diabetes Prediction'):
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies')
    with col2:
        Glucose = st.number_input('Glucose Level')
    with col3:
        BloodPressure = st.number_input('Blood Pressure Value')
    with col1:
        SkinThickness = st.number_input('Skin Thickness Value')
    with col2:
        Insulin = st.number_input('Insuline Level')
    with col3:
        BMI = st.number_input('BMI Value')
    with col1:
        DiabestesPredigreeFunction = st.number_input('Diabetes Pedigree Function Value')
    with col2:
        Age = st.number_input('Age of the Person')
    #code for prediction
    diab_diagnosis = ''
    inputX = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabestesPredigreeFunction,Age]
    input1 = transformVal(inputX)
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict(input1)

        if(diab_prediction[0]==1):
            diab_diagnosis = 'The person is Diabetic'
        else:
            diab_diagnosis = 'The person is Not Diabetic'
    st.success(diab_diagnosis)

if(selected == 'Heart Disease Prediction'):
    st.title('Heart Disease Prediction using ML')
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age')
    with col2:
        sex = st.number_input('Sex')
    with col3:
        cp = st.number_input('Chest Pain types')
    with col1:
        trestbps = st.number_input('Resting Blood Pressure')  
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl') 
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results') 
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.number_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy')
    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
    # code for Prediction
    heart_diagnosis = ''
    input2 = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
    input2 = transformVal(input2)

    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict(input2)                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)

if(selected == 'Parkinson Disease Prediction'):
    st.title('Parkinson Disease Prediction using ML')
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)') 
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)') 
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)') 
    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)') 
    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)')
    with col1:
        RAP = st.number_input('MDVP:RAP')
    with col2:
        PPQ = st.number_input('MDVP:PPQ')
    with col3:
        DDP = st.number_input('Jitter:DDP')
    with col4:
        Shimmer = st.number_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)')
    with col1:
        APQ3 = st.number_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.number_input('Shimmer:APQ5')
    with col3:
        APQ = st.number_input('MDVP:APQ')
    with col4:
        DDA = st.number_input('Shimmer:DDA')
    with col5:
        NHR = st.number_input('NHR')
    with col1:
        HNR = st.number_input('HNR')
    with col2:
        RPDE = st.number_input('RPDE')
    with col3:
        DFA = st.number_input('DFA')
    with col4:
        spread1 = st.number_input('spread1')
    with col5:
        spread2 = st.number_input('spread2')
    with col1:
        D2 = st.number_input('D2')
    with col2:
        PPE = st.number_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''
    input3 = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]
    input3 = transformVal(input3)
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinson_disease_model.predict(input3)                          
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)

if(selected == 'Breast Cancer Detection'):
    st.title('Breast Cancer Detection using ML')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        radius = st.number_input('Mean Radius')
    with col2:
        texture = st.number_input('Mean Texture')
    with col3:
        perimeter = st.number_input('Mean Perimeter')
    with col4:
        area = st.number_input('Mean Area')
    with col5:
        smoothness = st.number_input('Mean Smoothness')
    with col1:
        compactness = st.number_input('Mean Compactness')
    with col2:
        concavity = st.number_input('Mean Concavity')
    with col3:
        convpoints = st.number_input('Mean Concave Points')
    with col4:
        symmetry = st.number_input('Mean Symmetry')
    with col5:
        frdim = st.number_input('Mean Fractal Dimension')
    ####
    with col1:
        raderr =st.number_input('Radius Error')
    with col2:
        texterr = st.number_input('Texture Error')
    with col3:
        perierr = st.number_input('Perimeter Error')
    with col4:
        areaerr = st.number_input('Area Error')
    with col5:
        smoothnesserr = st.number_input('Smoothness Error')
    with col1:
        compactnesserr = st.number_input('Compactness Error')
    with col2:
        concavityerr = st.number_input('Concavity Error')
    with col3:
        convpointserr = st.number_input('Concave Points Error')
    with col4:
        symmetryerr = st.number_input('Symmetry Error')
    with col5:
        frdimerr = st.number_input('Fractal Dimension Error')
    ###
    with col1:
        wradius = st.number_input('Worst Radius')
    with col2:
        wtexture = st.number_input('Worst Texture')
    with col3:
        wperimeter = st.number_input('Worst Perimeter')
    with col4:
        warea = st.number_input('Worst Area')
    with col5:
        wsmoothness = st.number_input('Worst Smoothness')
    with col1:
        wcompactness = st.number_input('Worst Compactness')
    with col2:
        wconcavity = st.number_input('Worst Concavity')
    with col3:
        wconvpoints = st.number_input('Worst Concave Points')
    with col4:
        wsymmetry = st.number_input('Worst Symmetry')
    with col5:
        wfrdim = st.number_input('Worst Fractal Dimension')
    # code for Prediction
    breast_cancer_diagnosis = ''
    input4 = [radius,texture,perimeter,area,smoothness,compactness,concavity,convpoints,symmetry,frdim,
              raderr,texterr,perierr,areaerr,smoothnesserr,compactnesserr,concavityerr,convpointserr,symmetryerr,frdimerr,
              wradius,wtexture,wperimeter,warea,wsmoothness,wcompactness,wconcavity,wconvpoints,wsymmetry,wfrdim]
    input4 = transformVal(input4)
    # creating a button for Prediction    
    if st.button("Breast Cancer's Test Result"):
        breastcancer_prediction = model.predict(input4)  
        prediction_label = [np.argmax(breastcancer_prediction)]
        if (prediction_label[0] == 1):
          parkinsons_diagnosis = "The person has Breast Cancer"
        else:
          parkinsons_diagnosis = "The person does not have Breast Cancer"
        
    st.success(breast_cancer_diagnosis)


