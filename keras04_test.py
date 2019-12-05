from keras.models import Sequential
from keras.layers import Dense

import numpy as np  #numpy를 np로 줄인다
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

model = Sequential()
model.add(Dense(50, input_dim=1, activation='relu'))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1)     #fit = 기계를 트레이닝 시킴

loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print('acc : ', acc)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict)

