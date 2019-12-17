from keras.models import Sequential

filter_size = 32
kernel_size = (3,3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()
model.add(Conv2D(7, (2,2), # padding='same', 
                 input_shape=(28,28,1)))    # (27,27,7)

# model.add(Conv2D(16,(2,2)))
# model.add(MaxPooling2D(3,3))
# model.add(Conv2D(8,(2,2)))
model.add(Flatten())    # 27*27*7 = 5103
model.add(Dense(10))
model.add(Dense(10))

model.summary()