#!/usr/bin/env python
# coding: utf-8

# In[37]:


#Importing libraries

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error 


# In[38]:


energy_data = pd.read_csv(r"C:\Users\useer\Downloads\energydata_complete.csv")
energy_data.head()


# In[17]:


energy_data.info()


# In[18]:


energy_data.describe()


# In[19]:


energy_data.shape


# In[20]:


energy_data.isnull()


# In[24]:


#linear regression

X = energy_data['T2'].values.reshape(-1, 1)
y = energy_data['T6'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#Calculate RMSE
print(f"Root Mean Squared Error: {rmse:.3f}")


# In[39]:


# Set the target variable "Appliances"
X = energy_data.drop(columns=["date", "lights", "Appliances"])
y = energy_data["Appliances"]

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the dataset using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit a multiple linear regression model to the training set
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = model.predict(X_train)

# Calculate the Mean Absolute Error (MAE) for the training set
mae = mean_absolute_error(y_train, y_train_pred)

# Calculate the Mean Absolute Error (MAE) for the training set
mae = mean_absolute_error(y_train, y_train_pred)

# Print the MAE (rounded to three decimal places)
print(f"Mean Absolute Error for the training set: {mae:.3f}")


# In[ ]:





# In[ ]:




