#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


# In[2]:


data = np.genfromtxt("Edi_month_grid - Copy.csv", delimiter = ',')


# In[3]:


print(data)


# In[4]:


data.shape[0]


# In[5]:


data.shape[1]


# In[6]:


univariate_g1 = data[:,0]
train = univariate_g1[0:420]
test = univariate_g1[421:457]

print(train)


# In[7]:


train.shape[0]


# In[8]:


def rmse(pred, actual): 
	return np.sqrt(((pred-actual)**2).mean())


# In[9]:


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


# In[10]:


from numpy import array
n_steps = 3
x_train, y_train = split_sequence(train,n_steps)

# summarize the data
for i in range(len(x_train)):
	print(x_train[i], y_train[i])


# In[11]:


n_steps = 3
x_test, y_test = split_sequence(test,n_steps)

# summarize the data
for i in range(len(x_test)):
	print(x_test[i], y_test[i])


# In[12]:


mlp_adam = MLPRegressor(hidden_layer_sizes=(5, ), activation='relu', solver='adam', alpha=0.1,max_iter=5000, tol=0)
mlp_adam.fit(x_train,y_train)
y_predicttrain = mlp_adam.predict(x_train)
y_predicttest = mlp_adam.predict(x_test)
train_acc = rmse( y_predicttrain, y_train) 
test_acc = rmse( y_predicttest, y_test) 


# In[13]:


print(train_acc)


# In[14]:


print(test_acc)


# In[15]:


print(y_test)


# In[16]:


print(y_predicttest)


# In[17]:


#train
correlation_matrix = np.corrcoef(y_train, y_predicttrain)
correlation_xy = correlation_matrix[0,1]
r_squared_train = correlation_xy**2

print(r_squared)


# In[ ]:


#test
correlation_matrix = np.corrcoef(y_test, y_predicttest)
correlation_xy = correlation_matrix[0,1]
r_squared_test = correlation_xy**2

print(r_squared)


# In[ ]:


#from sklearn.metrics import r2_score
#TRAIN 
#train_r2 = r2_score(y_train, y_predicttrain)
test_r2 = r2_score(y_test, y_predicttest)

#print(train_r2, test_r2)


# In[18]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_train, y_predicttrain)


# In[19]:


mean_squared_error(y_test, y_predicttest)


# In[20]:


# plot expected vs predicted for training stage
from matplotlib import pyplot

pyplot.plot(y_train, label='Expected')
pyplot.plot(y_predicttrain, label='Predicted')
pyplot.legend()
pyplot.show()


# In[21]:


#plot for testing stage

pyplot.plot(y_test, label='Expected')
pyplot.plot(y_predicttest, label='Predicted')
pyplot.legend()
pyplot.show()


# In[22]:


x_train.shape


# In[23]:


y_train.shape


# In[24]:


mlp_sgd = MLPRegressor(hidden_layer_sizes=(5, ), activation='relu', solver='sgd', alpha=0.1,max_iter=5000, tol=0)
mlp_sgd.fit(x_train,y_train)
y_predicttrain = mlp_sgd.predict(x_train)
y_predicttest = mlp_sgd.predict(x_test)
train_acc = rmse( y_predicttrain,y_train) 
test_acc = rmse( y_predicttest, y_test) 

pyplot.plot(y_train, label='Expected')
pyplot.plot(y_predicttrain, label='Predicted')
pyplot.legend()
pyplot.show()



# In[25]:


#performance metrics # error
print(train_acc, test_acc)


# In[26]:


#train
correlation_matrix = np.corrcoef(y_train, y_predicttrain)
correlation_xy = correlation_matrix[0,1]
r_squared_train = correlation_xy**2

#test
correlation_matrix = np.corrcoef(y_test, y_predicttest)
correlation_xy = correlation_matrix[0,1]
r_squared_test = correlation_xy**2



print(r_squared_train, r_squared_test)


# In[27]:


print(y_predicttest)


# In[28]:


from sklearn.metrics import mean_squared_error
train_mse= mean_squared_error(y_train, y_predicttrain)

test_mse = mean_squared_error(y_test, y_predicttest)

print(train_mse,test_mse)


# In[29]:


# r2 value
from sklearn.metrics import r2_score
r2_train = r2_score(y_train, y_predicttrain)
r2_test = r2_score(y_test, y_predicttest)

print(r2_train, r2_test)


# In[49]:


def edi_cat(values):
    
    category = np.zeros(values.shape[0])
    
    print(category)
    
    for i in range(0, values.shape[0]):
        if values[i] >= 2:
            #msg[i] = "Extremely Wet"
            category[i] = 6
            
        elif values[i] > 1.5 and values[i] <= 1.99:
            #print("Very Wet")
            category[i] = 5
            
        elif values[i] > 1.0 and values[i] <= 1.49:
            #print("Moderately Wet")
            category[i] = 4
            
        elif values[i] > -0.99 and values[i] <= 0.99:
            #print("Normal")
            category[i] = 3
            
        elif values[i] > -1.49 and values[i] <= -1.0:
            #print("Moderately Dry")
            category[i] = 2
            
        elif values[i] > -1.99 and values[i] <= -1.5:
            #print("Severely Dry")
            category[i] = 1
            
        elif values[i] <= -2:
            #print("Extremely Dry")
            category[i] = 0
        else:
            #print(NA)
            
   # return  category 
            
    


# In[50]:


edi_cat(y_test)


# In[31]:


y_test.shape


# In[32]:


#validation categories
msg, category  = edi_cat(y_test)
predicted = edi_cat(y_predicttest)


# In[33]:


print(actual)


# In[59]:


def compare_listcomp(x, y):
    return [i for i, j in zip(x, y) if i == j]


# In[60]:


compare_listcomp(actual,predicted)


# In[21]:


#plot for testing stage

pyplot.plot(y_test, label='Expected')
pyplot.plot(y_predicttest, label='Predicted')
pyplot.legend()
pyplot.show()


# In[52]:


rf = RandomForestRegressor()
rf.fit(x_train,y_train)
y_predicttrain = rf.predict(x_train)
y_predicttest = rf.predict(x_test)
train_acc = rmse( y_predicttrain,y_train) 
test_acc = rmse( y_predicttest, y_test)

pyplot.plot(y_train, label='Expected')
pyplot.plot(y_predicttrain, label='Predicted')
pyplot.legend()
pyplot.show()


# In[53]:


#plot for testing stage

pyplot.plot(y_test, label='Expected')
pyplot.plot(y_predicttest, label='Predicted')
pyplot.legend()
pyplot.show()


# In[55]:


print(train_acc, test_acc)


# In[54]:


#train
correlation_matrix = np.corrcoef(y_train, y_predicttrain)
correlation_xy = correlation_matrix[0,1]
r_squared_train = correlation_xy**2

#test
correlation_matrix = np.corrcoef(y_test, y_predicttest)
correlation_xy = correlation_matrix[0,1]
r_squared_test = correlation_xy**2



print(r_squared_train, r_squared_test)


# In[56]:


from sklearn.metrics import mean_squared_error
train_mse= mean_squared_error(y_train, y_predicttrain)

test_mse = mean_squared_error(y_test, y_predicttest)

print(train_mse,test_mse)


# In[ ]:





# In[ ]:





# In[58]:


## keras example

from sklearn.datasets import make_circles
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot
 
from pandas import DataFrame


# In[59]:


###repeat same process above for one time step prediction
#define the model

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


# In[60]:


data = np.genfromtxt("Edi_month_grid - Copy.csv", delimiter = ',')
univariate_g1 = data[:,0]
train = univariate_g1[0:420]
test = univariate_g1[421:457]

print(train)


# In[61]:


from numpy import array
n_steps = 3

x_train, y_train = split_sequence(train,n_steps)

# summarize the data
for i in range(len(x_train)):
	print(x_train[i], y_train[i])
    
n_steps = 3

x_test, y_test = split_sequence(test,n_steps)

# summarize the data
for i in range(len(x_test)):
	print(x_test[i], y_test[i])


# In[62]:


n_features = 1
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
print(x_train.shape)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
print(x_test.shape)


# In[63]:


n_features = 1
# define model
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(x_train, y_train, epochs=400, verbose=0)


# In[65]:


#do the prediction 
yhat = model.predict(x_test, verbose = 0)
print(yhat)


# In[66]:


##plot and compare pred
pyplot.plot(yhat, label='Predicted')
pyplot.plot(y_test, label='Expected')
pyplot.legend()
pyplot.show()


# In[67]:


#train
correlation_matrix = np.corrcoef(y_train, yhat)
correlation_xy = correlation_matrix[0,1]
r_squared_train = correlation_xy**2

#test
correlation_matrix = np.corrcoef(y_test, y_predicttest)
correlation_xy = correlation_matrix[0,1]
r_squared_test = correlation_xy**2



print(r_squared_train, r_squared_test)


# In[ ]:





# In[ ]:





# In[15]:


def edi_cat(values):
    for i in range(0, y_test.shape[0]):
        if values[i] >= 2:
            print("Extremely Wet")
        elif values[i] > 1.5 and values[i] <= 1.99:
            print("Very Wet")
        elif values[i] > 1.0 and values[i] <= 1.49:
            print("Moderately Wet")
        elif values[i] > -0.99 and values[i] <= 0.999:
            print("Normal")
        elif values[i] > -1.49 and values[i] <= -1.0:
            print("Moderately Dry")
        elif values[i] > -1.99 and values[i] <= -1.5:
            print("Severely Dry")
        elif values[i] <= -2:
            print("Extremely Dry")
        else:
            print(NA)
    return 


# In[ ]:




