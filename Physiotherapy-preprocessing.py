#!/usr/bin/env python
# coding: utf-8

# In[162]:


import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
from sklearn.preprocessing import OneHotEncoder
import scipy
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,RocCurveDisplay,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


# In[163]:


data=pd.read_csv("E:/university/mashhad university/dr.amiri/revized data.csv")
#print("check for null",data.isnull().sum())
print(data.columns)
data.info()


# In[36]:


Columns=[ 'Sex', 'Pain', 'Painfulknee', 'DLL', 'Palpation',
       'Resisted.IC', 'Patellar.TT', 'Eeccentric.ST',"GS"]
plt.rcParams["figure.figsize"] = [20.00, 60]
plt.rcParams["figure.autolayout"] = True
f, axes = plt.subplots(5, 2)
i=0
for column in Columns :
    #sns.catplot(data=data, x=column, hue="CMPIE Status" , kind="count",ax=axes[i])
    sns_plot=sns.countplot(data=data,x=data[column].sort_values(), palette=sns.color_palette("Set2"), hue=data["GS" ],ax=axes[i//2,i%2]).set_title(f' Distribution of {column}')
    i=i+1


# In[38]:


Columns=['Age', 'BMI', 'Time',
        'Qangle', 'NavicularDT',
       'Torsion', 'Craigs', 'Squatting', 'Climbing', 'Desending', 'Kneeling',
       'Sitting']
plt.rcParams["figure.figsize"] = [20.00, 60]
plt.rcParams["figure.autolayout"] = True
f, axes = plt.subplots(6, 2)
i=0
for column in Columns:
   
    sns_plot=sns.histplot(data=data,x=data[column].sort_values(), palette=sns.color_palette("Set2"), hue=data["GS" ],ax=axes[i//2,i%2],kde=True).set_title(f' Distribution of {column}')
  
    i=i+1


# In[24]:


# Implementing a Kolmogorov Smirnov test in python scipy
from scipy.stats import kstest
Columns=['Age', 'BMI', 'Time',
        'Qangle', 'NavicularDT',
       'Torsion', 'Craigs', 'Squatting', 'Climbing', 'Desending', 'Kneeling',
       'Sitting']
datapfn=data[data["GS"]=="non,pfps"]
for i in Columns:
    print([i])
    stat,P=kstest(datapfn[i],"norm")
    print('Statistics=%.3f, p=%.3f' % (stat, P))
    alpha = 0.05
    if P > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')


# In[25]:


# Implementing a Kolmogorov Smirnov test in python scipy
from scipy.stats import kstest
Columns=['Age', 'BMI', 'Time',
        'Qangle', 'NavicularDT',
       'Torsion', 'Craigs', 'Squatting', 'Climbing', 'Desending', 'Kneeling',
       'Sitting']
datapf=data[data["GS"]=="pfps"]
for i in Columns:
    print([i])
    stat,P=kstest(datapf[i],"norm")
    print('Statistics=%.3f, p=%.3f' % (stat, P))
    alpha = 0.05
    if P > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')


# In[39]:


from scipy.stats import mannwhitneyu
Columns=['Age', 'BMI', 'Time',
        'Qangle', 'NavicularDT',
       'Torsion', 'Craigs', 'Squatting', 'Climbing', 'Desending', 'Kneeling',
       'Sitting']
datapf=data[data["GS"]=="pfps"]
datapfn=data[data["GS"]=="non,pfps"]
for i in Columns:
    print([i])
    stat,P=mannwhitneyu(datapf[i],datapfn[i],method="exact")
    print('Statistics=%.3f, p=%.3f' % (stat, P))
    alpha = 0.05
    if P > alpha:
        print('No significant relation (Accept  H0)')
    else:
        print('significant relation  (reject H0)')


# In[43]:


from scipy.stats import chi2_contingency
Columns=[ 'Sex', 'Pain', 'Painfulknee', 'DLL', 'Palpation',
       'Resisted.IC', 'Patellar.TT', 'Eeccentric.ST',"GS"]
results = {}
for col in Columns:
    print([col])
    contingency_table = pd.crosstab(data[col], data["GS"])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    results[col] = {"Test Statistic": chi2, "p-value": p}
    print('Statistics=%.3f, p=%.3f' % (chi2, p))
    alpha = 0.05
    if p > alpha:
        print('No significant relation (Accept  H0)')
    else:
        print('  significant relation(reject H0)')


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




