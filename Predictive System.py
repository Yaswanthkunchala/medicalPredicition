import numpy as np
import pickle


#loading the saved Model
loaded_model = pickle.load(open('./insurance_trained_model.sav', 'rb'))

input_data =(31,0,25.74, 0,0,0)

#changing input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)
