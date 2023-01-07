#!/usr/bin/env python
# coding: utf-8

# In[49]:


STUDENT_NAME = 'LohitPoojary' #Put your name
STUDENT_ROLLNO = 'MT2022139' #Put your roll number
CODE_COMPLETE = True


# In[ ]:


import numpy as np 
import pandas as pd 
import sklearn.model_selection as model_selection 
import sklearn.preprocessing as preprocessing 
import sklearn.metrics as metrics 
from tqdm import tqdm # You can make lovely progress bars using this


# In[51]:


X_train = pd.read_csv("train_X.csv",index_col=0).to_numpy()
y_train = pd.read_csv("train_y.csv",index_col=0).to_numpy().reshape(-1,)
X_test = pd.read_csv("test_X.csv",index_col=0).to_numpy()
submissions_df = pd.read_csv("sample_submission.csv",index_col=0)


# In[52]:


X_train = pd.DataFrame(X_train)
y_train=pd.DataFrame(y_train)
X_test=pd.DataFrame(X_test)


# In[53]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)


# In[54]:


s=set()
for i in (X_train.columns):
        s.add(X_train.dtypes[i])
print(s)


# In[55]:


X_train.duplicated().sum()


# In[56]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_trainm = scaler.fit_transform(X_train)
X_trainm = pd.DataFrame(X_trainm)


# In[57]:


X_trainm.describe()


# In[58]:


out=[]
for i in range(y_train.shape[0]):
    out.append(y_train.iloc[i][0])
X_trainm['lab']=out
X_trainm=X_trainm.sample(frac = 1)
train=X_trainm.to_numpy()
train


# In[59]:


from random import random
from random import seed
def initial_param(inputs_n,hidd_n,output_n):
    #Array consisting of all parameters
    params=[]
    
    #create layers
    layer1=[]
    for i in range(hidd_n):
        weigh={'w':[random() for i in range(inputs_n+1)]}
        layer1.append(weigh)
    layer2=[]
    for i in range(output_n):
        weigh={'w':[random() for i in range(hidd_n+1)]}
        layer2.append(weigh)
    params.append(layer1)
    params.append(layer2)
    
    return params


# In[60]:


def add(w,inp):
    b=w[-1]
    if(len(w)==len(inp)):
        wx=np.dot(w[:-1],inp[:-1])
    else:
        wx=np.dot(w[:-1],inp[:])
    return wx+b


# In[61]:


def lrel(x):
    return max(0,0.01*x)


# In[62]:


from numpy import exp
def soft(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# In[63]:


def ForwardPropagate(params,inputs):
    c=0
    for i in params:
        #print(i)
        # layer 1
        c+=1
        curr=[]
        for j in i:
            #neuron 1
            wxplusb=add(j['w'],inputs)
            
            if(c==2):
                j['out']=wxplusb
                curr.append(j['out'])
            else:
                j['out']=lrel(wxplusb)
                curr.append(j['out'])
        if(c==2):
            curr=soft(curr)
            k=0
            for j in i:
                j['out']=curr[k]
                k+=1
        
        inputs=curr
    return inputs
        
            


# In[64]:


def newweigh(params,inputs,alpha):
    for i in range(len(params)):
        x=inputs[:-1]
        if(i==0):
            for j in params[i]:
                for k in range(len(x)):
                    j['w'][k]-=alpha*(j['d'])*(x[k])
                j['w'][-1]-=alpha*(j['d'])*1
        else:
            x=[j['out']  for j in params[i-1]]
            for j in params[i]:
                for k in range(len(x)):
                    j['w'][k]-=alpha*(j['d'])*(x[k])
                j['w'][-1]-=alpha*(j['d'])*1


# In[65]:


def BackPropagate(params,Actual):
    for i in range(len(params)-1,-1,-1):
        curr_l=params[i]
        diff=[]
        if(i==len(params)-1):
            delt=0
            for j in range(len(curr_l)):
                diff.append(curr_l[j]['out']-Actual[j])
        else:
            for j in range(len(curr_l)):
                delt=0
                for k in params[i+1]:
                    delt+=k['d']*k['w'][j]
                diff.append(delt)
        for a in range(len(curr_l)):
            curr_l[a]['d']=curr_l[a]['out']*(1-curr_l[a]['out'])*(diff[a])


# In[66]:


def model(params,train,alpha,epochs,outputs):
    #params=initial_param(len(train[0])-1,50,outputs)
    for i in range(epochs):
        error_rate=0
        for inputs in train:
            #Forward propagate
            result=ForwardPropagate(params,inputs)
            
            #store actual results in an array 
            Actual=[0 for i in range(outputs)]
            #Set the required output as 1
            Actual[int(inputs[-1])]=1
            
            r=0
            for k in range(len(Actual)):
                r+=(Actual[k]-result[k])**2
            error_rate+=r
            
            #Backpropagate
            BackPropagate(params,Actual)
            newweigh(params,inputs,alpha)
        print('epoch',i,"alpha",alpha,'error',error_rate)
            


# In[67]:


from random import seed
seed(1)
params=initial_param(len(train[0])-1,270,10)


# In[68]:


len(train[0])-1


# In[69]:


model(params,train[:20001],0.0030,12,10)


# In[70]:


def fit(params,inp):
    out=ForwardPropagate(params,inp)
    a=list(out)
    return a.index(max(a))


# In[71]:


test=train[:1001]
c=0
for i in test:
    act=i[-1]
    out=fit(params,i)
    if(out==act):
        c+=1
print(c)


# In[72]:


out=[]
for i in range(X_test.shape[0]):
    out.append(0)
X_test['lab']=out
test=X_test.to_numpy()
test


# In[73]:


test=test
a=[]
for i in test:
    #act=i[-1]
    out=fit(params,i)
    a.append(int(out))
    


# In[74]:


X_test.drop('lab',axis=1,inplace=True)
X_test['label']=a


# In[75]:


submissions_df=X_test['label']


# In[76]:


submissions_df.to_csv("{}__{}.csv".format(STUDENT_ROLLNO,STUDENT_NAME))

