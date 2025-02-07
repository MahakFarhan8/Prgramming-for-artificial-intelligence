#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[2]:


train_data=pd.read_csv(r"C:\Users\Hamza Computer\Desktop\home-data-for-ml-course\train.csv")
train_data


# In[3]:


test_data=pd.read_csv(r"C:\Users\Hamza Computer\Desktop\home-data-for-ml-course\test.csv")
test_data


# In[4]:


train_data.info()


# In[5]:


y=train_data.SalePrice


# In[6]:


def dataEncoder(cols):
    for i in cols:
        dataLabelEncoder = LabelEncoder()
        train_data[i] = dataLabelEncoder.fit_transform(train_data[i])

columns = ['HouseStyle']
dataEncoder(columns)


# In[8]:


features = ['YearBuilt', 'HouseStyle']


# In[9]:


x = train_data[features]


# In[10]:


x.describe()


# In[11]:


model_svc = SVC()
model_svc.fit(x, y)

print(model_svc)


# In[12]:


def dataEncoder(cols):
    for i in cols:
        dataLabelEncoder = LabelEncoder()
        test_data[i] = dataLabelEncoder.fit_transform(test_data[i])

columns = ['HouseStyle']
dataEncoder(columns)


# In[13]:


features_test = ['YearBuilt', 'HouseStyle']


# In[14]:


o= test_data[features_test]


# In[15]:


df = pd.DataFrame(o)

# Save the DataFrame as a new CSV file
df.to_csv(r'test88.csv', index=False, header=True)

print("New CSV file created successfully!")


# In[16]:


test10_data=pd.read_csv('test88.csv')


# In[17]:


model_predictions = model_svc.predict(test10_data)


# In[18]:


x=pd.read_csv(r"C:\Users\Hamza Computer\Desktop\home-data-for-ml-course\test.csv")
x


# In[19]:


submission10=pd.DataFrame({'ID':x['Id'],'SalePrice':model_predictions})
submission10.to_csv('submission10.csv',index=False)
print("submission successfully")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




