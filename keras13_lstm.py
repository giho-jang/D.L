from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7])
print(x)
print('x.shape : ', x.shape)    #(4, 3)
print('y.shape : ', y.shape)    #(4, )

'''
 x  y
123 4
234 5
345 6
456 7
'''

x = x.reshape((x.shape[0], x.shape[1], 1))
print(x)
print('x.shape : ', x.shape)    #(4, 3, 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(3,1)))   # 3 = 컬럼, 1 = 1개씩 잘라서 작업할것인가
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(300))
model.add(Dense(1))
# model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=200, batch_size=1)

x_input = array([6,7,8])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)

#4. 평가예측