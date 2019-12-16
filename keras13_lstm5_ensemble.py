# x, y 데이터를 2개로 분리
# 2개의 인푸 모델인 ensemble 모델로 구현
# 수정해야함!

from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1 = x[:10]
x2 = x[10:]
y1 = y[:10]
y2 = y[10:]

print('x1 : ', x1)
print('x2 : ', x2)
print('y1 : ', y1)
print('y2 : ', y2)
print()

print('x1.shape : ', x1.shape)  #(10, 3)
print('x2.shape : ', x2.shape)  #(3, 3)
print('y1.shape : ', y1.shape)  #(10, )
print('y2.shape : ', y2.shape)  #(3, )

#2. 모델구성
input1 = Input(shape=(3,1))
lstm = LSTM(10)(input1)
dense1 = Dense(5, activation='relu')(lstm)
dense2 = Dense(4)(dense1)
dense3 = Dense(3)(dense2)
middle1 = Dense(3)(dense3)

input2 = Input(shape=(3,1))
lstm = LSTM(10)(input2)
xx = Dense(5, activation='relu')(lstm)
xx = Dense(4)(xx)
xx = Dense(3)(xx)
middle2 = Dense(3)(xx)

from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])    # 2개이상의 input, output이면 list

output1 = Dense(30)(merge1)
output1 = Dense(13)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(merge1)
output2 = Dense(31)(output2)
output2 = Dense(3)(output2)

model = Model(inputs = [input1, input2], outputs= [output1, output2])
model.summary()


#3. 실행
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='mse', patience=100, mode='auto')
model.fit(x, y, epochs=10000, callbacks=[early_stopping])    #early_stopping : 원하는곳에서 멈추게 해줌

x1_input = array([25,35,45]) # 1, 3, ??
x1_input = x1_input.reshape((1,3,1))

yhat = model.predict(x1_input)
print(yhat)
