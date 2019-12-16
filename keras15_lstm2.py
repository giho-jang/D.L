import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))

size = 5
def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print('====================')
print(dataset)

x_train = dataset[:, 0:-1]
y_train = dataset[:, -1]

print('x_train.shape : ', x_train.shape)    # (6,4)
print('y_train.shape : ', y_train.shape)    # (6, )

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

#2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(4, 1)))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
# model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=200, batch_size=1)

x2 = np.array([7,8,9,10])   # (4, ) -> (1, 4)
x2 = x2.reshape((1,4,1))

y_predict = model.predict(x2)
print(y_predict)
