#!/usr/bin/env python
# coding: utf-8

# # Diabetes prediction with Logistic Regression

# In[51]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale,StandardScaler # for standardization
from sklearn.model_selection import train_test_split, GridSearchCV ,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score, mean_squared_error, r2_score,classification_report,roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression


# In[ ]:


# turn off alerts
from warnings import filterwarnings
filterwarnings ('ignore')


# # Dataset and Story

# Purpose: There is some information about the people in the data set kept in our hospital. We are asked to perform a estimation model about whether the person has diabetes according to the results of the analysis.

# In[8]:


df= pd.read_csv("diabetes.csv")


# In[9]:


df.head()


# # Model and Prediction

# In[54]:


df["Outcome"].value_counts() # representation of the dependent variable.


# Veride 1 yani şeker hastası sayısında 268 adet kişinin bilgileri, 0 yani şeker hastası olmayan kişilerin verilerinden ise 500 kişinin bilgileri bulunmaktadır.

# In[10]:


df.describe().T # descriptive statistics


# In[56]:


y=df["Outcome"]# get dependent variable


# In[55]:


X=df.drop(["Outcome"], axis=1) # getting arguments


# In[15]:


loj_model=LogisticRegression(solver ="liblinear").fit(X,y)# model installed


# In[58]:


loj_model.intercept_ # fixed value of the model


# In[17]:


loj_model.coef_ # coefficients of arguments


# In[18]:


loj_model.predict(x)[0:10]


# In[19]:


y[0:10]


# In[57]:


y_pred = loj_model.predict(X) # predictive acquisition values


# In[21]:


confusion_matrix(y,y_pred) # confusion matrix


# In[24]:


accuracy_score(y,y_pred) # success rate


# In[29]:


print(classification_report(y,y_pred)) #detaylı raporlaması


# In[31]:


loj_model.predict_proba(X)[0:10] # gives the probability of classes.


# ### ROC 

# In[40]:


logit_roc_auc = roc_auc_score(y,loj_model.predict(X)) # grafik 
fpr,tpr,theresholds= roc_curve(y,loj_model.predict_proba(X)[:,1])#eğri
plt.figure() 
plt.plot(fpr,tpr,label='AUC (area= %0.2f)'  % logit_roc_auc)
plt.plot([0,1],[0,1],'r--')# eksen
plt.xlim([0.0,1.0])#eksen
plt.ylim([0.0,1.05])#eksen
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.legend ('Log_ROC')
plt.show()


# Comment: A graph that plots False-Positive rejection vs. True-Positive rejects to predict model success

# # Model Tuning

# In[41]:


X_train,X_test,y_train,y_test = train_test_split(X,# independent variable
                                                y,#the dependent variable
                                                test_size=0.30,# test data
                                                random_state=42) 


# In[43]:


loj_model = LogisticRegression(solver= "liblinear").fit(X_train,y_train)


# In[46]:


y_pred= loj_model.predict(X_test)


# In[47]:


print(accuracy_score(y_test,y_pred))


# In[53]:


cross_val_score(loj_model,X_test,y_test, cv=10).mean()


# In[ ]:




