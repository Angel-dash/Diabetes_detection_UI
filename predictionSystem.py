# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np 
import pickle

#loding the saved model 
loaded_model=pickle.load(open('C:/Users/Angel/Documents/Deploying_machine_learning/trained_model.sav','rb'))
input_data = (5,166,72,19,175,25.8,0.587,51)

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
if (prediction[0]==0):#The prediction gives the values the value in the form of a list and the result is displayed at an index of 0 so
  print("The person is not diabetic")
else:
    print("The person is diabetic")