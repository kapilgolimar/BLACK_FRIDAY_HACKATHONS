# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:37:11 2018

@author: kapil
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset_train = pd.read_csv('train.csv') 
dataset_train.fillna(999, inplace=True)
X = dataset_train.iloc[:,:-1].values
y = dataset_train.iloc[:,-1].values
x_df = pd.DataFrame(data=X,columns=['User_ID','Product_ID','Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3'])
y_df = pd.DataFrame(data=y,columns=['Purchase'])

dataset_test = pd.read_csv('test.csv')
dataset_test.fillna(999, inplace=True)
x_test_df = dataset_test 
        
X_data = x_df.append(x_test_df)    
X_data1 = X_data.iloc[:,:].values

for i in range(2,11):
    lbl = LabelEncoder()
    lbl.fit(X_data1[:,i])
    X_data1[:, i] = lbl.transform(X_data1[:, i])

X_data1 = X_data1[:,2:]
X_data1 = X_data1.astype(int)    

onehotencoder = OneHotEncoder(categorical_features = [1])
X_data1 = onehotencoder.fit_transform(X_data1).toarray()
X_data1 = np.delete(X_data1, 0, axis=1)

onehotencoder = OneHotEncoder(categorical_features = [8])
X_data1 = onehotencoder.fit_transform(X_data1).toarray()
X_data1 = np.delete(X_data1, 0, axis=1)

onehotencoder = OneHotEncoder(categorical_features = [10])
X_data1 = onehotencoder.fit_transform(X_data1).toarray()
X_data1 = np.delete(X_data1, 0, axis=1)

'''onehotencoder = OneHotEncoder(categorical_features = [29])
X_data = onehotencoder.fit_transform(X_data).toarray()
X_data = np.delete(X_data, 0, axis=1)

onehotencoder = OneHotEncoder(categorical_features = [34])
X_data = onehotencoder.fit_transform(X_data).toarray()
X_data = np.delete(X_data, 0, axis=1)

onehotencoder = OneHotEncoder(categorical_features = [53])
X_data = onehotencoder.fit_transform(X_data).toarray()
X_data = np.delete(X_data, 0, axis=1)

onehotencoder = OneHotEncoder(categorical_features = [70])
X_data = onehotencoder.fit_transform(X_data).toarray()
X_data = np.delete(X_data, 0, axis=1)'''

X = X_data1[:550068,:]
X_test = X_data1[550068:,:]
y = y_df.values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X= sc_X.fit_transform(X)
X_test = sc_X.transform(X_test)
y = sc_y.fit_transform(y)

#Predicting the Test set results
#import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim =16 ,kernel_initializer='uniform', activation = 'relu', input_dim = 18))
classifier.add(Dense(output_dim = 16, kernel_initializer='uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 16, kernel_initializer='uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, kernel_initializer='uniform'))
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['accuracy'])
classifier.fit(X, y, batch_size = 20, nb_epoch = 60)


y_pred = classifier.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)




