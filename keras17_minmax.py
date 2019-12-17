from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
           [20000,30000,40000], [30000,40000,50000],
           [40000,50000,60000], [100,200,300]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)   # 훈련시 가중치가 생김
x = scaler.transform(x)  #evaluate, predict

# train과 prodict로 나눌것
# train = 1번째부터 13번째까지
# predict = 14번째

x_train = x[:13]
x_predict = x[13:]
y_train = y[:13]
y_predict = y[13:]

print("x_train : ", x_train)
print("x_predict : ", x_predict)
print()
print('x.shape : ', x.shape)    #(14, 3)
print('y.shape : ', y.shape)    #(14, )

# x = x.reshape((x.shape[0], x.shape[1], 1))
# print('x.shape : ', x.shape)    #(4, 3, 1)

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(3, )))
model.add(Dense(90, activation='relu'))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1, verbose=2)    # verbose=0 : 훈련과정을 생략

import numpy as np

# predict용 데이터(예측)
# x_input = array([25,35,45]) # 1, 3, ????
# x_input = np.transpose(x_input)
# x_input = scaler.transform(x_input)

# print(x_input.shape)
# x_input = x_input.reshape((1,3))

yhat = model.predict(x_predict, verbose=2)
print(yhat)