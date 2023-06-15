#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 22:54:08 2022

@author: bunty
"""

import numpy as np 
import pickle            # for loading the saved model 
import streamlit as st   # for creating the web page

loaded_model = pickle.load(open('./insurance_trained_model.sav', 'rb'))

def medical_cost_prediction(input_data):
    

    #changing input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance 
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    return prediction

    
           
def main():
    
    
    # giving a title
    st.title('Medical Cost Prediction App ')  
    
    
    # getting the input data from the user
      
    age = st.text_input('Age of a person')
    sex = st.text_input('Sex of a person 0->female, 1->male')
    bmi = st.text_input('BMI of a person')
    children = st.text_input('Children of a person')
    smoker = st.text_input('Is a person smoker or not 1->yes or 0->No')
    region = st.text_input('Location : southeast->0,southwest->1 ,northeast->2, northwest->3')
   

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction

    if st.button('Medical Cost :'):
        diagnosis = medical_cost_prediction([age, sex, bmi, children, smoker, region ])
    
    
    st.success(diagnosis)



if __name__ == '__main__': 
    main() 


















     