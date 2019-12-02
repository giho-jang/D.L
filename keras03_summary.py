from keras.models import Sequential
from keras.layers import Dense

import numpy as np  #numpy를 np로 줄인다
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
x2 = np.array([11,12,13,14,15])

model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.fit(x, y, epochs=200)	#fit = 기계를 트레이닝 시킴, 

# loss, acc = model.evaluate(x, y)
# print('acc : ', acc)
# print('loss : ', loss)

# y_predict = model.predict(x2)
# print(y_predict)
