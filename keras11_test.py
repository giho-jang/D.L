#1. 데이터
import numpy as np  #numpy를 np로 줄인다
# x = np.array(range(1, 101))
# y = np.array(range(1, 101))
x = np.array([range(1, 101), range(101, 201)])
y = np.array([range(201, 301)])
# print(x)

print(x.shape)  #(2, 100)

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)  # (100, 2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=50, train_size=0.6, test_size=0.4, shuffle=False)
from sklearn.model_selection import train_test_split
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=50, test_size=0.5, shuffle=False)  # 6:2:2

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(50, input_dim=1, activation='relu'))
model.add(Dense(200, input_shape=(2, ), activation='relu'))
model.add(Dense(320))
model.add(Dense(650))
model.add(Dense(120))
model.add(Dense(470))
model.add(Dense(790))
model.add(Dense(830))
model.add(Dense(510))
model.add(Dense(990))
model.add(Dense(1))


# model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x, y, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))
 
#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse : ', mse)
print('loss : ', loss)

# aaa = np.array([[101,102,103],[201,202,203]])
# aaa = np.transpose(aaa)
# y_predict = model.predict(aaa)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   # sqrt = root(루트)
print('RMSE : ', RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)

# mse :  6164971.0
# loss :  6164970.925
# [[1955.545 ]
#  [2035.1239]
#  [2114.7017]
#  [2194.2803]
#  [2273.8582]
#  [2354.5425]
#  [2436.67  ]
#  [2518.7966]
#  [2600.9236]
#  [2683.051 ]
#  [2765.1775]
#  [2845.402 ]
#  [2926.8743]
#  [3014.4868]
#  [3102.0999]
#  [3185.6565]
#  [3267.475 ]
#  [3349.2932]
#  [3431.111 ]
#  [3518.6968]]
# RMSE :  2482.935898590913
# R2 :  -185411.65192515685