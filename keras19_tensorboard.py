#1. 데이터
import numpy as np  #numpy를 np로 줄인다
x = np.array(range(1, 101))
y = np.array(range(1, 101))
print(x)

# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]
# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=50, train_size=0.6, test_size=0.4, shuffle=False)
from sklearn.model_selection import train_test_split
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=50, test_size=0.5, shuffle=False)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
model = Sequential()

# model.add(Dense(50, input_dim=1, activation='relu'))
model.add(Dense(5, input_shape=(1, ), activation='relu'))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

import keras
tb_hist = keras.callbacks.TensorBoard(log_dir='graph',
                                      histogram_freq=0, write_grads=True, write_images=True)

from keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2,
          validation_data=(x_val, y_val), callbacks=[early_stopping, tb_hist])

#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse : ', mse)
print('loss : ', loss)

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
