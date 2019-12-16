# input1 output2

#1. 데이터
import numpy as np  #numpy를 np로 줄인다
x1 = np.array([range(100), range(311,411), range(100)])
y1 = np.array([range(501,601), range(711,811), range(100)])

y2 = np.array([range(501,601), range(711,811), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

print("x1 : ", x1.shape) # (100, 3)
print("y1 : ", y1.shape) # (100, 3)
print("x2 : ", y2.shape) # (100, 3)
print()

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, random_state=50, train_size=0.6, test_size=0.4, shuffle=False)
x1_val, x1_test, y1_val, y1_test = train_test_split(
    x1_test, y1_test, random_state=50, test_size=0.5, shuffle=False)

y2_train, y2_test = train_test_split(
    y2, random_state=50, train_size=0.6, test_size=0.4, shuffle=False)
y2_val, y2_test = train_test_split(
    y2_test, random_state=50, test_size=0.5, shuffle=False)

print("y2_test : ", y2_test.shape)  # (20, 3)

#2. 모델구성(함수형모델)
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(3)(dense2)
middle1 = Dense(3)(dense3)

# from keras.layers.merge import concatenate
# merge1 = concatenate([middle1, middle2])    # 2개이상의 input, output이면 list

output1 = Dense(30)(middle1)
output1 = Dense(13)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(middle1)
output2 = Dense(31)(output2)
output2 = Dense(3)(output2)

model = Model(inputs = input1, outputs= [output1, output2])
model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x, y, epochs=100, batch_size=1)
model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=1,
          validation_data=(x1_val, [y1_val, y2_val]))
      
#4. 평가 예측
mse = model.evaluate(x1_test, [y1_test, y2_test], batch_size=1)
print('mse : ', mse)

y1_predict, y2_predict  = model.predict(x1_test)
print(y1_predict)
print()
print(y2_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(xxx, yyy):
    return np.sqrt(mean_squared_error(xxx, yyy))
RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", (RMSE1 + RMSE2)/2)

# R2 구하기
from sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, y1_predict)
r2_y2_predict = r2_score(y2_test, y2_predict)
print('R2_1 : ', r2_y1_predict)
print('R2_2 : ', r2_y2_predict)
print('R2 : ', (r2_y1_predict + r2_y2_predict)/2)