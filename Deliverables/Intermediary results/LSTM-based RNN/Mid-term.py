#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from matplotlib import style
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
from tensorflow.keras.layers import Dense, Dropout
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_excel("RomaniaEDF.xlsx")
df=df.dropna()
df["Datetime"]=pd.to_datetime(df[["Year", "Month", "Day", "Hour"]])
del df["Year"]
del df["Month"]
del df["Day"]
del df["Hour"]


# In[3]:


dataset = df
dataset["Month"] = pd.to_datetime(df["Datetime"]).dt.month
dataset["Year"] = pd.to_datetime(df["Datetime"]).dt.year
dataset["Date"] = pd.to_datetime(df["Datetime"]).dt.date
dataset["Time"] = pd.to_datetime(df["Datetime"]).dt.time
dataset["Week"] = pd.to_datetime(df["Datetime"]).dt.week
dataset["Day"] = pd.to_datetime(df["Datetime"]).dt.day_name()
dataset = df.set_index("Datetime")
dataset.index = pd.to_datetime(dataset.index)
dataset.head(5)


# In[4]:


plt.style.use('fivethirtyeight')
fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

style.use('ggplot')

sns.lineplot(x=dataset["Year"], y=dataset["Value"], data=df)
sns.set(rc={'figure.figsize':(15,6)})

plt.title("Energy consumptionnin Year 2004")
plt.xlabel("Date")
plt.ylabel("Energy in MW")
plt.grid(True)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)


plt.title("Energy Consumption According to Year")


# In[6]:


plt.style.use('fivethirtyeight')

fig = plt.figure()


ax2= fig.add_subplot(312)
ax3= fig.add_subplot(313)


style.use('ggplot')

y_2015 = dataset["2015"]["Value"].to_list()
x_2015 = dataset["2015"]["Date"].to_list()
ax2.plot(x_2015, y_2015, color="green", linewidth=1)


y_2006 = dataset["2006"]["Value"].to_list()
x_2006 = dataset["2006"]["Date"].to_list()
ax3.plot(x_2006, y_2006, color="red", linewidth=1)


plt.rcParams["figure.figsize"] = (18,8)
plt.xlabel("Date")
plt.ylabel("Energy in MW")
plt.grid(True, alpha=1)
plt.legend()

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(90)


# In[7]:


plt.style.use('fivethirtyeight')

sns.distplot(dataset["Value"])
plt.title("Ennergy Distribution")


# In[8]:


NewDataSet = dataset.resample('D').median()
NewDataSet = NewDataSet.dropna()


# In[9]:


NewDataSet


# In[10]:


TestData = NewDataSet.tail(100)

Training_Set = NewDataSet.iloc[:,0:1]

Training_Set = Training_Set[:-60]

Training_Set = Training_Set
sc = MinMaxScaler(feature_range=(0, 1))
Train = sc.fit_transform(Training_Set)


# In[11]:


X_Train = []
Y_Train = []


for i in range(60, Train.shape[0]):
    
    # X_Train 0-59 
    X_Train.append(Train[i-60:i])

    Y_Train.append(Train[i])

# Convert into Numpy Array
X_Train = np.array(X_Train)
Y_Train = np.array(Y_Train)

print(X_Train.shape)
print(Y_Train.shape)


# In[12]:


X_Train = np.reshape(X_Train, newshape=(X_Train.shape[0], X_Train.shape[1], 1))
X_Train.shape


# In[14]:


regressor = Sequential()


regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_Train.shape[1], 1)))
regressor.add(Dropout(0.1))


regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_Train.shape[1], 1)))
regressor.add(Dropout(0.1))


regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.1))


regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[15]:


regressor.fit(X_Train, Y_Train, epochs = 25, batch_size = 12)


# In[16]:


Df_Total = pd.concat((NewDataSet[["Value"]], TestData[["Value"]]), axis=0)


# In[17]:


inputs = Df_Total[len(Df_Total) - len(TestData) - 60:].values
inputs.shape


# In[18]:


inputs = Df_Total[len(Df_Total) - len(TestData) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 160):
    X_test.append(inputs[i-60:i])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

 
predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# In[19]:


True_MegaWatt = TestData["Value"].to_list()
Predicted_MegaWatt  = predicted_stock_price
dates = TestData.index.to_list()


# In[20]:


Machine_Df = pd.DataFrame(data={
    "Date":dates,
    "TrueMegaWatt": True_MegaWatt,
    "PredictedMeagWatt":[x[0] for x in Predicted_MegaWatt ]
})


# In[21]:


Machine_Df


# In[22]:


True_MegaWatt = TestData["Value"].to_list()
Predicted_MegaWatt  = [x[0] for x in Predicted_MegaWatt ]
dates = TestData.index.to_list()


# In[23]:


plt.style.use('seaborn-pastel')
fig = plt.figure()

ax1= fig.add_subplot(111)

x = dates
y = True_MegaWatt

y1 = Predicted_MegaWatt

plt.plot(x,y, color="green")
plt.plot(x,y1, color="red")
# beautify the x-labels
plt.gcf().autofmt_xdate()
plt.xlabel('Dates')
plt.ylabel("Power in MW")
plt.title("EDF with LSTM ")
plt.legend()


# In[24]:


Machine_Df.head(10)


# In[25]:


def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) 


# In[26]:


mape(Machine_Df[["TrueMegaWatt"]],Machine_Df[["PredictedMeagWatt"]])


# In[33]:


from sklearn.metrics import mean_squared_error
import math


# In[36]:


MSE = mean_squared_error(Machine_Df[["TrueMegaWatt"]],Machine_Df[["PredictedMeagWatt"]])
 
RMSE = math.sqrt(MSE)

RMSE


# In[30]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(Machine_Df[["TrueMegaWatt"]],Machine_Df[["PredictedMeagWatt"]])

