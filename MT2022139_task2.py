#!/usr/bin/env python
# coding: utf-8

# In[48]:


STUDENT_ROLLNO = 'MT2022139' 
STUDENT_NAME = 'LohitPoojary'


# In[1]:


import torch


import torch.nn as nn 
import torch.optim as optim 
import torchmetrics

import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection 
from torch.utils.data import Dataset, DataLoader
import numpy as np 


# In[2]:


X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
submission = np.load("sample_submission.npy")


# In[3]:


X_train.shape


# In[4]:


y_train.shape


# In[5]:


y_train


# In[6]:


X_train1=X_train[:30001]
y_train1=y_train[:30001]
X_test1=X_train[30001:]
y_test1=y_train[30001:]


# In[7]:


X_train2=X_train1.reshape(-1,X_train1.shape[1])
X_train2=X_train2.astype('float32')


# In[8]:


X_test2=X_test1.reshape(-1,X_test1.shape[1])
X_test2=X_test2.astype('float32')
X_test2=torch.from_numpy(X_test2)
y_test1=torch.from_numpy(y_test1)


# In[69]:


class Fun(Dataset):
    
    def __init__(self):
        self.x_train=torch.from_numpy(X_train2)
        self.y_train=torch.from_numpy(y_train1)
        
        self.len=self.x_train.shape[0]
        
        
    def __getitem__(self,p):
        return self.x_train[p],self.y_train[p]
    
    
    def __len__(self):
        return self.len


# In[70]:


train=DataLoader(dataset=Fun(),batch_size=64)


# In[71]:


class create_network(nn.Module):
    def __init__(self,input_n,hidd_n,hidd_n2,output):
        super(create_network,self).__init__()
        self.layer1=nn.Linear(input_n,hidd_n)
        self.layer2=nn.Linear(hidd_n,hidd_n2)
        self.layer3=nn.Linear(hidd_n2,output)
    def forward(self,var):
        var=torch.relu(self.layer1(var))
        var=torch.relu(self.layer2(var))
        var=self.layer3(var)
        
        return var


# In[72]:


model=create_network(input_n=3072,hidd_n=1500,hidd_n2=1000,output=10)
losscri=nn.CrossEntropyLoss()
op=torch.optim.SGD(model.parameters(),lr=0.0001)


# In[73]:


def model_train(epochs):
    for i in range(epochs):
        for j,k in train:
            op.zero_grad()
            m=model(j)
            l=losscri(m,k)
            l.backward()
            op.step()


# In[74]:


model_train(60)


# In[43]:


X_test=X_test.reshape(-1,X_test.shape[1])
X_test=X_test.astype('float32')
X_test=torch.from_numpy(X_test)


# In[44]:


p=model(X_test)
y_pred=torch.max(p.data,1)
y_pred[1]


# In[51]:


np.save("{}__{}".format(STUDENT_ROLLNO,STUDENT_NAME),y_pred[1])


# In[ ]:




