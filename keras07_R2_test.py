from keras.models import Sequential
from keras.layers import Dense

import numpy as np  #numpy를 np로 줄인다
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
# x_predict = np.array([21,22,23,24,25])

model = Sequential()
# model.add(Dense(50, input_dim=1, activation='relu'))
model.add(Dense(500, input_shape=(1, ), activation='relu'))
model.add(Dense(492))
model.add(Dense(215))
model.add(Dense(252))
model.add(Dense(428))
model.add(Dense(269))
model.add(Dense(151))
model.add(Dense(456))
model.add(Dense(351))
model.add(Dense(311))
model.add(Dense(320))
model.add(Dense(395))
model.add(Dense(412))
model.add(Dense(272))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam',
              # metrics=['accuracy'])
              metrics=['mse'])
model.fit(x_train, y_train, epochs=200, batch_size=1)   # fit = 기계를 트레이닝 시킴

loss, mse = model.evaluate(x_test, y_test, batch_size=1)    # a[0], a[1]
print('mse : ', mse)    # 1.0 / 1.153158700617496e-05
print('loss : ', loss)  # 1.1039837772841566e-07 / 1.153158700617496e-05

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))   # sqrt = root(루트)
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print('R2 : ', r2_y_predict)

# 문제1. R2를 0.5이하로 줄이시오.
# 레이어는 인푸와 아웃푸 포함 5개이상, 노드는 각 레이어당 5개 이상
# batch_size = 1
# epochs = 100이상

# mse :  84.20380401611328
# loss :  84.20379753112793
# [[ 1.3295974]
#  [ 1.063919 ]
#  [ 2.5391693]
#  [ 4.0121994]
#  [ 5.4888763]
#  [ 6.9639664]
#  [ 8.438171 ]
#  [ 9.914185 ]
#  [11.389885 ]
#  [12.864845 ]]
# RMSE :  9.17562355908051
# R2 :  -9.205099114903472
