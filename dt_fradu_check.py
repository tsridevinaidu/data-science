#!/usr/bin/env python
# coding: utf-8

# In[90]:


import numpy as np


# In[91]:


import pandas as pd


# In[92]:


import matplotlib.pyplot as plt


# In[93]:


fraud=pd.read_csv("C:\\Users\\Sridevi T\\Downloads\\sridevi\\Private\\Private\\Locker\\Fraud_check.csv")


# In[94]:


fraud


# In[95]:


fraud.sort_values('Urban', inplace=True)


# In[96]:


fraud


# In[97]:


d1=fraud[fraud["Urban"]=="YES"]


# In[98]:


d1 


# In[99]:


d2=fraud[fraud["Urban"]=="NO"]


# In[100]:


d2


# In[101]:


df1 = pd.DataFrame(d1.iloc[0:25, 0:6])


# In[102]:


df1


# In[103]:


df2 = pd.DataFrame(d2.iloc[0:25, 0:6])


# In[104]:


df2


# In[105]:


fraud_train = pd.concat([df1,df2])
fraud_train


# In[106]:


x_train=fraud_train.iloc[:, ~fraud_train.columns.isin(['Undergrad', 'Marital.Status', 'Urban'])]


# In[107]:


x_train


# In[108]:


y_train=fraud_train.iloc[:,5]


# In[109]:


y_train


# In[110]:


tf1=pd.DataFrame(d1.iloc[26:, 0:6])


# In[111]:


tf1


# In[112]:


tf2=pd.DataFrame(d2.iloc[26:, 0:6])


# In[113]:


tf2


# In[114]:


fraud_test=pd.concat([tf1, tf2])
fraud_test


# In[115]:


x_test=fraud_test.iloc[:, ~fraud_test.columns.isin(['Undergrad', 'Marital.Status', 'Urban'])]
x_test


# In[ ]:





# In[116]:


y_test=fraud_test.iloc[:, 5]
y_test


# In[123]:


# Fitting Decision Tree Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)


# In[124]:


# Predicting the Test set results
y_pred = classifier.predict(x_test)


# In[125]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[126]:


cm


# In[127]:


from sklearn.metrics import accuracy_score 
Accuracy_Score = accuracy_score(y_test, y_pred)


# In[128]:


Accuracy_Score


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




