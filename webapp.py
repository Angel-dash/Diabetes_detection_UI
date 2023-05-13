# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:00:41 2023

@author: Angel
"""

import numpy as np
import pickle #to load a trained model 
import streamlit as st #used to create a webpage 
loaded_model=pickle.load(open('C:/Users/Angel/Documents/Deploying_machine_learning/trained_model.sav','rb'))
#rb means reading a file in abinary format
#creating a function for predection 

#taking input data from the user and converting it to numpy array
def diabetes_prediction(input_data):
    #change the input data to a numpy array 
    input_data_as_numpy_array=np.asarray(input_data)

    #reshape the array as we area predecting for one instance
    #if we do not reshape the array then the model will expect to have 764 data which is the total number of dataset
    #it will tell the model we are not looking for all the other 764 instances but only for one instances
    input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

    #we need to standarized the input data since we are not using the raw data but the standarized input data

    #standaraizes the input data
    #std_data=scaler.transform(input_data_reshaped)
    #print(std_data)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] ==0 ):#The prediction gives the values the value in the form of a list and the result is displayed at an index of 0 so
        return "The person is not diabetic"
    else:
        return "The person is diabetic"
    
#creating a user intrface    
def main():
    #title 
    st.title("Diabetes prediction WEB APP")
    
    #getting the input data from the user
    
    Pregnancies=st.text_input("Enter the number of pregnancies")
    Glucose=st.text_input("Glucose level")
    BloodPressure=st.text_input("Blood pressure levels")
    SkinThickness=st.text_input("Enter skinthickness")
    Insulin=st.text_input("Enter the level of insulin ")
    BMI=st.text_input("Enter BMI of the patience")
    DiabetesPedigreeFunction=st.text_input("Enter the diabetes pedegree function ")
    Age=st.text_input("Enter Age of the patinet")
    
    #code for prediction.This is were the final results will be stored
    diagnosis=''
    
    #creating a button for prediction 
    if st.button('Diabetes test result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose, BloodPressure,
                                       SkinThickness,Insulin, BMI,DiabetesPedigreeFunction, Age])
    st.success(diagnosis)
    
if __name__ == '__main__':#run only the main function, while running from anaconda
    main()
