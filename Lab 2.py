#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[72]:


train_df = pd.read_csv(r"C:\Users\Hamza Computer\Desktop\spaceship-titanic\train.csv")
train_df


# In[73]:


train_df.head()


# In[74]:


train_df.isnull().sum()


# In[75]:


train_df.info()


# In[76]:


def fillNaObjMode(cols):
    for i in cols:
        train_df[i] = train_df[i].fillna(train_df[i].mode()[0])

columns = ['HomePlanet','CryoSleep','Cabin','Destination','VIP','Name']
fillNaObjMode(columns)


# In[77]:


def fillNaFloat(cols):
    for i in cols:
        train_df[i] = train_df[i].fillna(train_df[i].mean())

columns = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
fillNaFloat(columns)


# In[78]:


def convertFloatintoInt(cols):
    for i in cols:
        train_df[i] = train_df[i].astype('int64')

columns =  ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
convertFloatintoInt(columns)


# In[79]:


def dataEncoder(cols):
    for i in cols:
        dataLabelEncoder = LabelEncoder()
        train_df[i] = dataLabelEncoder.fit_transform(train_df[i])

columns = ['PassengerId','HomePlanet','CryoSleep','Cabin','Destination','VIP','Name']
dataEncoder(columns)


# In[80]:


X_train = train_df.drop(['Transported', 'PassengerId'], axis=1)
y_train = train_df['Transported']


# In[81]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# In[82]:


test_df = pd.read_csv(r"C:\Users\Hamza Computer\Desktop\spaceship-titanic\test.csv")
test_df


# In[83]:


test_df.head()


# In[84]:


test_df.info()


# In[85]:


test_df.isnull().sum()


# In[86]:


def fillNaObjMode(cols):
    for i in cols:
        test_df[i] = test_df[i].fillna(test_df[i].mode()[0])

columns = ['PassengerId','HomePlanet','CryoSleep','Cabin','Destination','VIP','Name']
fillNaObjMode(columns)


# In[87]:


def fillNaFloat(cols):
    for i in cols:
        test_df[i] = test_df[i].fillna(test_df[i].mean())

columns = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
fillNaFloat(columns)


# In[88]:


def convertFloatintoInt(cols):
    for i in cols:
        test_df[i] = test_df[i].astype('int64')

columns =  ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
convertFloatintoInt(columns)


# In[89]:


feature_test = test_df.drop(['PassengerId'],axis=1)


# In[90]:


def dataEncoder(cols):
    for i in cols:
        dataLabelEncoder = LabelEncoder()
        test_df[i] = dataLabelEncoder.fit_transform(test_df[i])

columns = ['PassengerId','HomePlanet','CryoSleep','Cabin','Destination','VIP','Name']
dataEncoder(columns)


# In[91]:


feature_test = test_df.drop(['PassengerId'],axis=1)


# In[93]:


model_predictions = model.predict(feature_test)


# In[94]:


x = pd.read_csv(r"C:\Users\Hamza Computer\Desktop\spaceship-titanic\test.csv")


# In[95]:


submission1=pd.DataFrame({'PassengerId':x['PassengerId'],'Transported':model_predictions})
submission1.to_csv('submission1.csv',index=False)
print("submission successfully")


# In[ ]:




