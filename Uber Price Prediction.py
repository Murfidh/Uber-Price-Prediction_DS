#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression


# In[83]:


#load the data
ride_df = pd.read_csv('C:/Users/moham/Downloads/cab_prices/cab_rides.csv')
weather_df = pd.read_csv('C:/Users/moham/Downloads/cab_prices//weather.csv')


# In[84]:


ride_df 


# In[85]:


weather_df


# In[86]:


ride_df.info()


# In[87]:


weather_df.info()


# In[10]:


#Cleaning ride data 


# In[88]:


ride_df


# In[89]:


ride_df.isna().sum()


# In[90]:


ride_df = ride_df.dropna(axis=0).reset_index(drop=True)


# In[15]:


#Cleaning wether data


# In[91]:


weather_df


# In[92]:


weather_df.isna().sum()


# In[93]:


weather_df = weather_df.fillna(0)


# In[19]:


#Creating average wether  dataFrame


# In[94]:


weather_df.groupby('location').mean()


# In[95]:


avg_weather_df =weather_df.groupby('location').mean().reset_index(drop=False)
avg_weather_df = avg_weather_df.drop('time_stamp',axis=1)
avg_weather_df


# In[ ]:


#Merging DataFarames


# In[96]:


ride_df


# In[38]:


#including wether for the source and destination  


# In[97]:


source_weather_df = avg_weather_df.rename(
    columns={
             'location' : 'source',
              'temp' : 'source_temp',  
              'clouds' : 'source_clouds',
              'pressure' : 'source_pressure',
              'rain' : 'source_rain',
              'humidity' : 'source_humidity',
              'wind' : 'source_wind'
    }
)

source_weather_df


# In[98]:


destination_weather_df = avg_weather_df.rename(
    columns={
             'location' : 'destination',
              'temp' : 'destination_temp',  
              'clouds' : 'destination_clouds',
              'pressure' : 'destination_pressure',
              'rain' : 'destination_rain',
              'humidity' : 'destination_humidity',
              'wind' : 'destination_wind'
            }
)

destination_weather_df


# In[99]:


data = ride_df    .merge(source_weather_df, on='source')    .merge(destination_weather_df, on='destination')

data


# In[100]:


#preprocessing 
 


# In[114]:


def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df


# In[121]:


def preprocess_inputs(df):
    df = df.copy()
    
    # Drop id column
    df = df.drop('id', axis=1)
    
    # Binary encode cab_type column
    df['cab_type'] = df['cab_type'].replace({'Lyft': 0, 'Uber': 1})
    
    # One-hot encode remaining categorical columns
    for column, prefix in [('destination', "dest"), ('source', "src"), ('product_id', "pid"), ('name', "nm")]:
        df = onehot_encode(df, column=column, prefix=prefix)
    
    # Split df into X and y
    y = df['price']
    X = df.drop('price', axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    
    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    return X_train, X_test, y_train, y_test


# In[123]:


X_train, X_test, y_train, y_test = preprocess_inputs(data)


# In[107]:


{column:len(X[column].unique()) for column in X.select_dtypes('object').columns} #checking uniqe val is string


# In[124]:


X_train


# In[125]:


y_train


# In[126]:


#training 


# In[127]:


model = LinearRegression()
model.fit(X_train,y_train)

print("Test R-squred Score : {:.5f}".format(model.score(X_test,y_test)))

