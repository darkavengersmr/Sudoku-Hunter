#! /usr/bin/python3

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
import numpy as np
import os, time, cv2
from cv2 import imread, resize
from PIL import Image
import matplotlib.pyplot as plt

img_rows, img_cols = 64, 64
img_channels = 1
X = []
Y = []

for floder in range(9):
    for file in range(405):
        image = imread("./number/"+str(floder+1) + "/" + str(file+1)+".jpg", 0)
        image = cv2.resize(image, (img_rows, img_cols))
        max_value = np.max(image)
        min_value = np.min(image)
        image = (image - min_value) * (255/(max_value - min_value))
        cv2.imwrite("/home/ml/MLSudoku/img"+str(floder)+str(file)+".jpg", image)
        X.append(image)
        Y.append(floder)
    print("load - ", floder+1)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15)
X_train, X_testv, y_train, y_testv = train_test_split(X_train, y_train, test_size=0.1)

'''plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
plt.show()'''

X_testv = X_testv.reshape(X_testv.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)

input_shape = (1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_testv = X_testv.astype('float32')

X_train /= 255.0
X_test /= 255.0
X_testv /= 255.0

num_category = 9
y_train = to_categorical(y_train, num_category)
y_test = to_categorical(y_test, num_category)
y_testv = to_categorical(y_testv, num_category)

model = Sequential()

model.add(Conv2D(12, kernel_size=(3, 3), activation='relu', input_shape=input_shape, data_format = 'channels_first'))
model.add(Conv2D(32, kernel_size=(5, 5), activation='tanh', input_shape=input_shape, data_format = 'channels_first'))
model.add(Conv2D(32, kernel_size=(8, 8), activation='relu', input_shape=input_shape, data_format = 'channels_first'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1024, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(num_category, activation='softmax'))

model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])

batch_size = 12
num_epoch = 100
fail_epoch = 0
last_progress = 0
score = [0, 0]

for i in range(num_epoch):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=1, validation_data=(X_test, y_test))
    last_progress = score[1]
    score = model.evaluate(X_test, y_test, verbose=0)
    print('\n\n\nТочность на тестовых данных( эпоха -', i+1, '): %f2\n\n\n' % (score[1]*100))
    if(score[1]*100 > 95):
        print("Успешное обучение!")
        break
    if(last_progress > score[1]): fail_epoch += 1
    else: fail_epoch = 0
    if(fail_epoch >= 3): break

score = model.evaluate(X_testv, y_testv, verbose=0)
print('\n\n\nТочность на валидных данных: %f2\n\n\n' % (score[1]*100))

model_digit_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_digit_json)
model.save_weights("model.h5")
print("Saved model to disk")
