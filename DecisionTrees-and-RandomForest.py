#!/usr/bin/env python
# coding: utf-8

# 
# # Decision Trees and Random Forests in Python

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('kyphosis.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# ## EDA
# 
# We'll just check out a simple pairplot for this small dataset.

# In[34]:


sns.displot(x=df['Age'],hue=df['Kyphosis'])


# In[35]:


sns.displot(x=df['Age'])


# In[36]:


present=df[df['Kyphosis']=='present']


# In[37]:


sns.displot(present['Age'],color='DarkSlateGrey')


# In[38]:


absent=df[df['Kyphosis']=='absent']


# In[39]:


sns.displot(absent['Age'],bins=7,color='red')


# In[40]:


g = sns.FacetGrid(data=df,col='Kyphosis')
g.map(plt.hist,'Age')


# ## Train Test Split
# 
# Let's split up the data into a training set and a test set!

# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


X = df.drop('Kyphosis',axis=1)
y = df['Kyphosis']


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# ## Decision Trees
# 
# We'll start just by training a single decision tree.

# In[44]:


from sklearn.tree import DecisionTreeClassifier


# In[45]:


dtree = DecisionTreeClassifier()


# In[46]:


dtree.fit(X_train,y_train)


# ## Prediction and Evaluation 
# 
# Let's evaluate our decision tree.

# In[47]:


predictions = dtree.predict(X_test)


# In[48]:


from sklearn.metrics import classification_report,confusion_matrix


# In[49]:


print(classification_report(y_test,predictions))


# In[50]:


print(confusion_matrix(y_test,predictions))


# In[51]:


len(y_test)


# In[52]:


sum(y_test=='absent')


# In[53]:


sum(y_test=='present')


# ## Random Forests
# 
# Now let's compare the decision tree model to a random forest.

# In[54]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[55]:


rfc_pred = rfc.predict(X_test)


# In[56]:


print(confusion_matrix(y_test,rfc_pred))


# In[57]:


print(classification_report(y_test,rfc_pred))


# ## Logistic Regression model
# 
# Now let's compare the decision tree model and random forest to Logistic Regression model.

# In[58]:


from sklearn.linear_model import LogisticRegression


# In[59]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[60]:


predictions = logmodel.predict(X_test)


# In[61]:


confusion_matrix(y_test,predictions)


# In[62]:


print(classification_report(y_test,predictions))

