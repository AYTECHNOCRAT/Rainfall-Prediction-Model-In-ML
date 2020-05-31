#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


rainfall_data=pd.read_csv(r'/Volumes/Aditya Yadav/Study Material/ML and AI/rainfall-in-india/rainfall in india 1901-2015.csv')


# In[3]:


rainfall_data.head(4)


# In[4]:


rainfall_data.isnull().sum()


# In[5]:


rainfall_data=rainfall_data.fillna(rainfall_data.mean())
rainfall_data.isnull().sum()


# In[6]:


rainfall_data.hist(figsize=(12,12));


# In[7]:


rainfall_data.groupby("YEAR").sum()['ANNUAL'].plot(figsize=(12,8));


# In[8]:


rainfall_data[['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("YEAR").sum().plot(figsize=(13,8));


# In[10]:


rainfall_data[['YEAR','Jan-Feb','Mar-May','Jun-Sep','Oct-Dec','ANNUAL']].groupby("YEAR").sum().plot(figsize=(13,8));


# In[9]:


rainfall_data[['Jan-Feb','Mar-May','Jun-Sep','Oct-Dec','ANNUAL']].corr()


# In[10]:


rainfall_data[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']].corr()


# In[51]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

division_data = np.asarray(rainfall_data[['JAN','FEB', 'MAR', 'APR', 'MAY', 
                                          'JUN', 'JUL','AUG', 'SEP', 'OCT', 'NOV', 'DEC']])
n_times=division_data.shape[1]-3
X=None;
y=None;
for i in range(division_data.shape[1]-3):
    if X is None:
        X = division_data[:, i:i+3]
        y = division_data[:, i+3]
    else:
        X = np.concatenate((X, division_data[:, i:i+3]), axis=0)
        y = np.concatenate((y, division_data[:, i+3]), axis=0)
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[12]:


from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

# Linear model
lmodel = linear_model.ElasticNet(alpha=0.5)
lmodel.fit(X_train, y_train)
y_pred = lmodel.predict(X_test)
print (mean_absolute_error(y_test, y_pred))


# In[16]:


lmodel.score(X_train,y_train)


# In[17]:


lmodel.score(X_test,y_test)


# In[34]:


from sklearn.tree import DecisionTreeRegressor

#DecisionTree Model
clf=DecisionTreeRegressor()
clf.fit(X_train.astype('float32'),y_train.astype('float32'))
y_pred=clf.predict(X_test)
print(mean_absolute_error(y_test, y_pred))


# In[35]:


clf.score(X_train,y_train)


# In[36]:


clf.score(X_test,y_test)


# In[37]:


from sklearn.svm import SVR

#SVM model
svm = SVR(gamma='auto', C=0.1, epsilon=0.2)
svm.fit(X_train, y_train) 
y_pred = svm.predict(X_test)
print (mean_absolute_error(y_test, y_pred))


# In[38]:


svm.score(X_train,y_train)


# In[39]:


svm.score(X_test,y_test)


# In[52]:


from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten
from tensorflow.keras.models import Model

#NN model
inputs = Input(shape=(3,1))
x = Conv1D(64, 2, padding='same', activation='elu')(inputs)
x = Conv1D(128, 2, padding='same', activation='elu')(x)
x = Flatten()(x)
x = Dense(128, activation='elu')(x)
x = Dense(64, activation='elu')(x)
x = Dense(32, activation='elu')(x)
x = Dense(1, activation='linear')(x)
model = Model(inputs=[inputs], outputs=[x])
model.compile(loss='mean_squared_error', optimizer='adamax', metrics=['mae'])


# In[56]:


model.fit(x=np.expand_dims(X_train, axis=2), y=y_train, batch_size=64, epochs=5,
          verbose=1, validation_split=0.1, shuffle=True)
y_pred = model.predict(np.expand_dims(X_test, axis=2))
print (mean_absolute_error(y_test, y_pred))


# In[ ]:


#END

