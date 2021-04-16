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

data = np.genfromtxt("Edi_month_grid - Copy.csv", delimiter=',')
univariate_g1 = data[:, 0]
train = univariate_g1[0:420]
test = univariate_g1[421:457]

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
    Best_RMSE = 1000  # Assigning a large number

    start_time = time.time()
    for run in range(Num_Exp):
        print("Experiment", run + 1, "in progress")
        # fit model
        model.fit(x_train, y_train, epochs=Epochs, batch_size=10, verbose=0, shuffle=False)

        y_predicttrain = model.predict(x_train)
        y_predicttest = model.predict(x_test)
        # print(y_predicttest)
        train_acc[run] = rmse(y_predicttrain, y_train)
        test_acc[run] = rmse(y_predicttest, y_test)
        if test_acc[run] < Best_RMSE:
            Best_RMSE = test_acc[run]
            Best_Predict_Test = y_predicttest
        for j in range(n_steps_out):
            Step_RMSE[run][j] = rmse(y_predicttest[:, j], y_test[:, j])

        chain_inp = []
        chain_out = []
        chain_inp.append(list(future_predict_df.tail(1).iloc[0, 0:6]))
        chain_out.append(list(future_predict_df.tail(1).iloc[0, 6:10]))
        chain_inp = np.asarray(chain_inp, dtype=np.float32)
        chain_out = np.asarray(chain_out, dtype=np.float32)
        results = []
        # for step in range (1,16):
        # chain_inp = np.concatenate([chain_inp.reshape(chain_inp.shape[0],chain_inp.shape[1],n_features)[:,-2:,:],chain_out.reshape(chain_out.shape[0],chain_out.shape[1],n_features)],axis=1)
        # chain_out = model.predict(chain_inp)
        # print(chain_out.shape)
        # for pred in chain_out[0]:
        # results.append(pred)
        # future_prediction[run][:] = np.ndarray.flatten(scaler.inverse_transform(np.reshape(results,(len(results),1))))
        # print(future_prediction)
    print("Total time for", Num_Exp, "experiments", time.time() - start_time)
    return future_prediction, train_acc, test_acc, Step_RMSE, Best_Predict_Test, y_predicttrain, y_predicttest


#------------------


n_steps_in = 3
n_steps_out = 3
x_train, y_train = split_sequence(train, n_steps_in, n_steps_out)

# summarize the data 
 
x_test, y_test = split_sequence(test, n_steps_in, n_steps_out)
 


Hidden = 10
Epochs = 20
n_steps_out = 3
n_steps_in = 3
name = 'Grid1'
Num_Exp = 3

future_prediction, train_acc, test_acc, Step_RMSE, Best_Predict_Test, y_predicttrain, y_predicttest = MODEL_LSTM(name,x_train,x_test,y_train,y_test,Num_Exp,n_steps_in,n_steps_out,Epochs, Hidden)

print(train_acc, test_acc) 