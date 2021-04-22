import copy
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import datetime
import time
import joblib
from datetime import timedelta, date
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from numpy import array


#print(train)


# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)




def rmse(pred, actual):
	return np.sqrt(((pred - actual) ** 2).mean())


# define the model

def MODEL_LSTM(name, x_train, x_test, y_train, y_test, Num_Exp, n_steps_in, n_steps_out, Epochs, Hidden):
	n_features = 1
	x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
	print(x_train.shape)
	x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
	print(x_test.shape)

	train_acc = np.zeros(Num_Exp)
	test_acc = np.zeros(Num_Exp)
	Step_RMSE = np.zeros([Num_Exp, n_steps_out])

	model = Sequential()
	model.add(LSTM(Hidden, activation='relu', input_shape=(n_steps_in, n_features), dropout=0.2))
	model.add(Dense(32))
	model.add(Dense(n_steps_out))
	model.compile(optimizer='adam', loss='mse')
	model.summary()
	future_prediction = np.zeros([Num_Exp, 60])


	y_predicttest_allruns = np.zeros([Num_Exp, x_test.shape[0], x_test.shape[1]])

	print(y_predicttest_allruns.shape, ' shape ')


	Best_RMSE = 1000  # Assigning a large number

	start_time = time.time()
	for run in range(Num_Exp):
		print("Experiment", run + 1, "in progress")
		# fit model
		model.fit(x_train, y_train, epochs=Epochs, batch_size=10, verbose=0, shuffle=False)

		y_predicttrain = model.predict(x_train)
		y_predicttest = model.predict(x_test)
		y_predicttest_allruns[run,:,:] = y_predicttest
		#print(y_predicttest)
		train_acc[run] = rmse(y_predicttrain, y_train)
		#print(train_acc[run])

		test_acc[run] = rmse(y_predicttest, y_test)
		if test_acc[run] < Best_RMSE:
			Best_RMSE = test_acc[run]
			Best_Predict_Test = y_predicttest
		for j in range(n_steps_out):
			Step_RMSE[run][j] = rmse(y_predicttest[:, j], y_test[:, j])

		#chain_inp = []
		#chain_out = []
		#chain_inp.append(list(future_predict_df.tail(1).iloc[0, 0:6]))
		#chain_out.append(list(future_predict_df.tail(1).iloc[0, 6:10]))
		#chain_inp = np.asarray(chain_inp, dtype=np.float32)
		#chain_out = np.asarray(chain_out, dtype=np.float32)
		#results = []
		# for step in range (1,16):
		# chain_inp = np.concatenate([chain_inp.reshape(chain_inp.shape[0],chain_inp.shape[1],n_features)[:,-2:,:],chain_out.reshape(chain_out.shape[0],chain_out.shape[1],n_features)],axis=1)
		# chain_out = model.predict(chain_inp)
		# print(chain_out.shape)
		# for pred in chain_out[0]:
		# results.append(pred)
		# future_prediction[run][:] = np.ndarray.flatten(scaler.inverse_transform(np.reshape(results,(len(results),1))))
		# print(future_prediction)
	print("Total time for", Num_Exp, "experiments", time.time() - start_time)
	return future_prediction, train_acc, test_acc, Step_RMSE, Best_Predict_Test, y_predicttrain, y_predicttest, y_predicttest_allruns

def edi_cat(values):
	
	category = np.zeros((values.shape[0], values.shape[1]))
	
	
	for i in range(0, values.shape[0]):

		for j in range(0, values.shape[1]):
			if values[i][j] >= 2: 
				category[i][j] = 6
				
			elif values[i][j] > 1.5 and values[i][j] <= 1.99: 
				category[i][j] = 5
				
			elif values[i][j] > 1.0 and values[i][j] <= 1.49: 
				category[i][j] = 4
				
			elif values[i][j] > -0.99 and values[i][j] <= 0.99: 
				category[i][j] = 3
				
			elif values[i][j] > -1.49 and values[i][j] <= -1.0: 
				category[i][j] = 2
				
			elif values[i][j] > -1.99 and values[i][j] <= -1.5: 
				category[i][j] = 1
				
			elif values[i][j] <= -2: 
				category[i][j] = 0
			else:
				category[i][j] = 7
	#print(category)
	#print(values)

	return  category 


#------------------MAIN--------------------

num_grids = 2

for grid in range(num_grids):

	print(grid, ' is grid')

	data = np.genfromtxt("Edi_month_grid - Copy.csv", delimiter=',')
	univariate_g1 = data[:, grid]
	train = univariate_g1[0:420]
	test = univariate_g1[421:457]


	n_steps_in = 3
	n_steps_out = 3
	x_train, y_train = split_sequence(train, n_steps_in, n_steps_out)

	# summarize the data 
	 
	x_test, y_test = split_sequence(test, n_steps_in, n_steps_out)
	 


	Hidden = 10
	Epochs = 50
	n_steps_out = 3
	n_steps_in = 3
	name = 'Grid1'
	Num_Exp = 2

	future_prediction, train_acc, test_acc, Step_RMSE, Best_Predict_Test, y_predicttrain, y_predicttest, y_predicttest_allruns = MODEL_LSTM(name,x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Epochs, Hidden)

	

	print(train_acc, test_acc) 

	mean_train = np.mean(train_acc, axis=0)
	mean_test = np.mean(test_acc, axis=0)
	std_train = np.std(train_acc, axis=0)
	std_test = np.std(test_acc, axis=0)

	step_rmse_mean = np.mean(Step_RMSE, axis=0)
	step_rmse_std = np.std(Step_RMSE, axis=0)


	print(mean_train, 'mean_train') 
	print(mean_test, 'mean_test') 

	print(std_train, 'std_train') 
	print(std_test, 'std_test') 


	print(step_rmse_mean, ' step_rmse mean') 
	print(step_rmse_std, ' step_rmse std') 



 


	y_predicttest_mean = np.mean(y_predicttest_allruns, axis=0)


	category_steps = edi_cat(y_predicttest_mean)


	print(category_steps)


	y_predicttest_std = np.std(y_predicttest_allruns, axis=0)

	y_predicttest_meanstd = np.concatenate((y_predicttest_mean, y_predicttest_std), axis=1)

	results_combined = np.array([mean_train, std_train, mean_test, std_test, step_rmse_mean, step_rmse_std])

	print(results_combined, ' results_combined ')

 


	#np.savetxt('results/results_summary_'+str(grid)+'_.txt', results_combined)

	np.savetxt('results/category_steps_'+str(grid)+'_.txt', category_steps)

	np.savetxt('results/y_predicttest_meanstd_'+str(grid)+'_.txt', y_predicttest_meanstd) 