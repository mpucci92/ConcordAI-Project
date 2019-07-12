#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import sys, os
from haversine import haversine
import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime, timedelta


# ### Importing data

# In[2]:


raw = pd.read_csv("C:\\Python\\Hackathon\\Internship\\coordinates-20_Users.csv")


# ### Format Data 

# In[3]:


### Head of the data
raw.head()


# ### Column Manipulation

# In[4]:


# Column types 
raw.dtypes


# In[5]:


list_ =[]
for val in raw.timestamp_txt:
    val1 = val.split("T")
    list_.append(val1)

time_ = []
for i in range(len(list_)):
    time = list_[i][1]
    time_.append(time)

clock_ = []
for time in time_:
    time = time.split("-")
    clock_.append(time)

clock1_ = []
for i in range(len(clock_)):
    clock = clock_[i][0]
    clock1_.append(clock)

clock = []
for i in clock1_:
    val = i[0:8]
    clock.append(val)


# In[6]:


raw['clock_time'] = pd.to_datetime(clock)


# In[7]:


raw['clock'] = raw.clock_time.dt.time


# In[8]:


raw['clock_date'] = pd.to_datetime(raw.timestamp)


# In[9]:


# Sanity Check 1
raw.head()


# In[35]:


raw.dtypes


# ### Montreal Weather Data 

# In[18]:


weather_df = pd.read_csv("C:\\Itinerum GIS DATA\\Data\\Weather_mtl.csv",skiprows=24,encoding="latin_1") # Skip the first 24 lines in order to get rid of legend


# In[19]:


weather_df.head()


# In[32]:


weather_df.columns

#filter_weatherdf = weather_df.drop(['Year', 'Month', 'Day', 'Data Quality','Max Temp Flag','Min Temp Flag','Mean Temp Flag', 'Heat Deg Days (°C)','Heat Deg Days Flag','Cool Deg Days (°C)', 'Cool Deg Days Flag','Total Rain Flag', 'Total Snow (cm)', 'Total Snow Flag', 'Total Precip Flag', 'Snow on Grnd (cm)','Snow on Grnd Flag', 'Dir of Max Gust (10s deg)','Dir of Max Gust Flag', 'Spd of Max Gust (km/h)','Spd of Max Gust Flag'],axis=1)


# In[33]:


weather_df.rename()


# In[37]:


weather_df.rename(columns={'Date/Time':'timestamp'},inplace=True)


# In[38]:


raw_weather = raw.merge(weather_df, on='timestamp', how='outer')


# In[39]:





# In[ ]:





# ### Sort values based on the trip_id and the timestamp 

# In[12]:


raw = raw.sort_values(['trip_id','timestamp']) 


# ### Convert to datetime - Generate Acceleration and Jerk Features 

# In[13]:


raw.timestamp = pd.to_datetime(raw.timestamp)


# In[14]:


raw['timediff'] = raw.timestamp.diff().dt.seconds + 1   #Time difference is in seconds


# In[15]:


raw['speeddiff'] = raw.speed.diff()                     # This will subtract the speed in postiion 2 - position 1


# In[16]:


raw['acceleration'] = raw.speeddiff / raw.timediff     # divide the speed difference with the time difference to get the acceleration 


# In[17]:


raw['accelerationdiff'] = raw.acceleration.diff()     # This will subtract the acceleration in position 2 - position 1 


# In[18]:


raw['jerk'] = raw.accelerationdiff / raw.timediff    # Jerk (4th derivative of Transportation)  


# ### Filtering the trip_id columns for values greater than 3

# In[19]:


raw = raw.groupby("trip_id").filter(lambda x: len(x) >= 3)


# In[118]:


raw.dtypes


# ### Hour Column Manipulation after the filtering of trip_ids less than 3

# In[21]:


raw['Hour'] = (raw.clock_time.dt.hour)


# In[22]:


sample = raw.groupby('trip_id').size()
unique_id = sample.to_dict()             # type(sample) # turn this into a dictionary 


# In[23]:


# Dictionary 
# unique_id


# ### Converting the coordinate list into a float numpy array 

# In[24]:


coordinates =  list(zip(raw.latitude, raw.longitude))
raw['LL'] = coordinates 


# ### Function to estimate the time of the day

# In[25]:


def hours(x):
    if x>=6 and x<=9:
        return 'MR'             # Morning Rush
    elif x>=15 and x<=18:
        return 'NR'             #Night rush
    else:
        return 'S' #Stationary
    
        


# In[26]:


raw['RushhourType'] = raw['Hour'].apply(hours)


# ### PublicTransportation closing hours

# In[27]:


def closed_transit(x):
    if x >=1 and x<=5:
        return 1
    else:
        return 0


# In[28]:


raw['ClosedTransit'] = raw['Hour'].apply(closed_transit)


# ### Distance Calculations

# In[29]:


vals = [np.nan]
LL_col = pd.DataFrame()
LL_col['LL'] = raw.LL

print (LL_col['LL'][0])
    


# In[30]:


vals = [np.nan]
LL_col = pd.DataFrame()
LL_col['LL'] = raw.LL.as_matrix()
print (haversine(LL_col['LL'][0],LL_col['LL'][1]))

    


# In[31]:


vals = [np.nan]
LL_col = pd.DataFrame()
LL_col['LL'] = raw.LL.as_matrix()

for i in range(1, raw.shape[0]):
    
    pair1 = LL_col['LL'][i-1]
    pair2 = LL_col['LL'][i]
    val = (haversine(pair1,pair2,unit='km'))
    vals.append(val)


# ### Total Distance Feature

# In[32]:


raw['Distance'] = vals


# ### Weekday and Weekends 

# In[33]:


raw['Weekends'] = raw.timestamp.dt.day_name()
raw.Weekends = raw.Weekends.replace(to_replace=['Saturday','Sunday'], value=1)
raw.Weekends = raw.Weekends.replace(to_replace=['Monday','Tuesday','Wednesday','Thursday','Friday'], value=0)
raw.Weekends.value_counts()


# ### Time of Day

# In[34]:


def time_of_day(x):
    if x>= 5 and x<10:
        return 'morning'
    elif x>=10 and x<12:
        return 'lunch'
    elif x>=12 and x<16:
        return 'afternoon'
    elif x>=16 and x<18:
        return 'evening'
    elif ((x>=18 and x<=24) or (x>24 and x<5)):
        return 'night'
    else:
        return np.nan

raw['TimeOfDay'] = raw.Hour.apply(time_of_day)


# ### Rough Features 

# In[35]:


# Percentile function for the aggregation

def pct(x):
    return np.quantile(x,0.85)


# In[36]:


raw.speed = raw.speed.astype(float)
groupby_list = ['trip_id']
temp = raw.groupby(groupby_list).agg({
    'speed' : [
        ('SpeedPct', pct),
        ('MaxSpeed', 'max'),
        ('STDSpeed', 'std'),
        ('MeanSpeed', 'mean')],
         'Weekends' : ['first'],
         'RushhourType': ['first'],
         'ClosedTransit' : ['first'],
         'TimeOfDay' : ['first'],
        'tc_50m' : ['first'],
        'train_50m':['first'],
        'infra_50m':['first']
})


# In[37]:


temp.head()


# In[38]:


test = raw.groupby(groupby_list).sum()

df_temp = pd.merge(temp,test['Distance'], on="trip_id") # Dataframe of the trades taken by the model 


# In[ ]:





# In[39]:


def distance(x):
    if len(x) <= 1:
        return 0
    else:
        return np.sum(x[1:])


# In[40]:


temp['TotalDistance'] = raw.groupby(groupby_list).apply(lambda x: distance(x['Distance'].values))


# In[41]:


tester = []

# Accessing the values of the dictionary with a loop 

for value in unique_id.values():
    tester.append(value)


temp['AverageDistanceBetweenPoints']= temp.TotalDistance/tester

    #   x = unique_id.values()
  #  val = distance/x
   # print(x)


# In[42]:


def pct(x):
    if len(x) <= 1:
        return 0
    else:
        return np.quantile(x[1:], 0.85)


# In[43]:


temp['AccelPct'] = raw.groupby(groupby_list).apply(lambda x: pct(x['acceleration'].values))


# In[44]:


def max_(x):
    if len(x) <= 1:
        return 0
    else:
        return max(x[1:])


# In[45]:


temp['MaxAccel'] = raw.groupby(groupby_list).apply(lambda x: max_(x['acceleration'].values))


# In[46]:


def STD(x):
    if len(x) <= 1:
        return 0
    else:
        return np.std(x[1:])


# In[47]:


temp['STDAccel'] = raw.groupby(groupby_list).apply(lambda x: STD(x['acceleration'].values))
temp['test'] = raw.groupby(groupby_list).apply(lambda x: STD(x['speed'].values))


# In[48]:


def mean(x):
    if len(x) <= 1:
        return 0
    else:
        return np.mean(x[1:])


# In[49]:


temp['MeanAccel'] = raw.groupby(groupby_list).apply(lambda x: mean(x['acceleration'].values))


# In[50]:


temp.head()


# In[51]:


temp['JerkPct'] = raw.groupby(groupby_list).apply(lambda x: pct(x['jerk'].values))


# In[52]:


temp['MaxJerk'] = raw.groupby(groupby_list).apply(lambda x: max_(x['jerk'].values))


# In[53]:


temp['STDJerk'] = raw.groupby(groupby_list).apply(lambda x: STD(x['jerk'].values))


# In[54]:


temp['MeanJerk'] = raw.groupby(groupby_list).apply(lambda x: mean(x['jerk'].values))


# In[55]:


temp.columns = ['SpeedPct', 'MaxSpeed', 'STDSpeed', 'MeanSpeed', 'Weekends', 'RushHourType', 'ClosedTransit', 'TimeOfDay', 'tc_50m','train_50m','infra_50m','TotalDistance','AverageDistanceBetweenPoints','AccelPct', 'MaxAccel', 'STDAccel','test','MeanAccel','JerkPct','MaxJerk','STDJerk','MeanJerk']


# In[56]:


temp.head()


# ### Reset Index and output to a CSV

# In[57]:


# This is done after the features have been created, need a column to aggregate the data, once this is done we can reset the index. 

temp.reset_index().to_csv('Rough_Featues.csv',index=False)


# In[58]:


temp.head()


# In[59]:


len(temp.dtypes)


# ### Prepping the Features 

# In[60]:


df_features = pd.read_csv('C:\\Python\\Hackathon\\Internship\\Rough_Featues.csv')


# In[61]:


sns.heatmap(df_features.notnull())  # Need to see why so many Nan Values in the STD Speed category


# In[62]:


df_features.fillna(value = 0, inplace=True)


# In[63]:


sns.heatmap(df_features.notnull())


# In[64]:


df_features.describe()


# In[65]:


df_features.drop(['test'],axis=1, inplace=True)


# In[66]:


from sklearn.preprocessing import StandardScaler


# In[67]:


df_features.columns


# In[68]:


features_to_scale = ['SpeedPct', 'MaxSpeed', 'STDSpeed', 'MeanSpeed', 'AccelPct', 'MaxAccel', 'STDAccel', 'MeanAccel', 'TotalDistance', 'AverageDistanceBetweenPoints','JerkPct','MaxJerk','STDJerk','MeanJerk']


# In[69]:


scaling_initialization = StandardScaler()
df_features.loc[:, features_to_scale] = scaling_initialization.fit_transform(df_features[features_to_scale])


# In[70]:


df_features_copy = df_features.copy()
copy_df_features = df_features.copy()


# ### OHE Function

# In[71]:


def OHEencode(df, cols):
    dummies = pd.get_dummies(df, columns = cols)
    output = pd.concat([original_dataframe, dummies], axis=1)
    return(output)


# In[72]:


OHE_features = ['TimeOfDay','RushHourType']


# In[73]:


features_finale = pd.get_dummies(copy_df_features,OHE_features)


# In[74]:


features_finale.columns


# In[75]:


test = pd.read_csv("C:\\Python\\Hackathon\\Internship\\Modes_20.csv")
test.rename(columns={'id':'trip_id'},inplace=True)
test.drop(['index'],axis=1,inplace=True)


# In[108]:


test['mode'].value_counts()


# In[110]:


len(test['mode'])
print('Frequency of 3')
print(107/195)
print(62/195)
print(18/195)
print(4/195)
print(2/195)
print(1/195)
print(1/195)


# In[76]:


final_features = features_finale.merge(test, on='trip_id', how='outer').dropna()


# In[77]:


final_features.iloc[:,1:-1].head()


# In[ ]:





# In[78]:


final_features.reset_index().to_csv('Final_Features.csv',index=False)


# In[79]:


### Can use RandomForest, AdaBoost, Bagging classifier, 


# ### TRAINING THE MODEL - Section 2

# In[80]:


import lightgbm as lgbm

import pandas as pd
import numpy as np

import sys, os

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score


# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

def confusion(true,pred):
    confusion_matrix = ConfusionMatrix(true, pred)
    confusion_matrix.plot()
    plt.set_cmap('Greens')
    plt.rcParams.update({'font.size': 12})
    return plt.show()


# ### Import the Data

# In[82]:


df = pd.read_csv('C:\\Python\\Hackathon\\Internship\\Final_Features.csv').iloc[:, 1:]


# ### Testing and Training sets 

# In[83]:


X_train = df.iloc[:, 1:-1]
y_train = df.iloc[:, -1]


# In[84]:


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)


# ### RandomForest Model

# In[85]:


rf = RandomForestClassifier()


# In[111]:


# 9.5 minutes to gridSearch the Parameters 
params = {
    'n_estimators' : [2,4,5,6,7,8,9,10,20],
    'max_depth' : [5, 10, 15, 20,25],
    'min_samples_leaf' : [5,10,15,20,25,30],
}


# In[112]:


gcv = GridSearchCV(rf, params, verbose=2)


# In[113]:


gcv.fit(X_train, y_train)


# ### Best Parameters

# In[114]:


gcv.best_params_


# In[115]:


gcv.best_score_


# In[117]:


# Input the best params in the following block of code - i randomly set these parameters for now. 
rf = RandomForestClassifier(max_depth=5, min_samples_leaf=5, n_estimators=10)
rf = rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_pred, y_test)


# ### Exporting the Randomforest

# In[92]:


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


# In[93]:


X_train.columns


# In[94]:


confusion(y_pred, y_test)


# 

# In[ ]:




