# from keras.datasets import mnist
 
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print(train_labels.shape)

# CNN code https://ithelp.ithome.com.tw/articles/10197257

from google.colab import drive
drive.mount('/content/drive')
import os
import re
import numpy as np
# please change your directory
print(os.listdir("/content/drive/MyDrive/Colab Notebooks/mnist/"))

import csv

with open("/content/drive/MyDrive/Colab Notebooks/mnist/emnist-letters-train.csv", newline='') as csvfile:
  rows = csv.reader(csvfile)
  print(type(rows))
  data = np.array([row for row in rows])

print(np.shape(data))
print(data.shape)
train = data[1:35001]
test = data[35001:]
train_labels = train[:, 0]
test_labels = test[:, 0]
train_images = train[:, 1:]
test_images = test[:, 1:]
print(train_labels.shape)
print(train_images.shape)
print(test_images.shape)
print(test_images.shape)

fix_train_images = train_images.reshape((35000, 28 , 28)).astype('float32') / 255
fix_test_images = test_images.reshape((7000, 28 , 28)).astype('float32') / 255

from keras.utils import np_utils
 
fix_train_labels = np_utils.to_categorical(train_labels)
fix_test_labels = np_utils.to_categorical(test_labels)

# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/random_shift
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/random_rotation
import keras.preprocessing.image
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
rotated = fix_train_images[:][:][:]
label_copy = fix_train_labels[:][:]
for i in range(4):
  rotated = np.concatenate([rotated, keras.preprocessing.image.random_shift(
      fix_train_images,
      0.2,
      0.2,    # 兩方向各20度
      row_axis=1,
      col_axis=2,
      channel_axis=0,
      fill_mode='nearest',
      cval=0.0,
      interpolation_order=1
  )])
  fix_train_labels = np.concatenate([fix_train_labels, label_copy])
  rotated = np.concatenate([rotated, keras.preprocessing.image.random_rotation(
      fix_train_images,
      40,    # 兩方向各20度
      row_axis=1,
      col_axis=2,
      channel_axis=0,
      fill_mode='nearest',
      cval=0.0,
      interpolation_order=1
  )])
  fix_train_labels = np.concatenate([fix_train_labels, label_copy])
fix_train_images = rotated
print(fix_train_images.shape)
print(fix_train_labels.shape)

# import matplotlib.pyplot as plt 

# # 建立函數要來畫多圖的
# def plot_images_labels_prediction(images,labels,prediction,idx,num=10): 
  
#   # 設定顯示圖形的大小
#   fig= plt.gcf()
#   fig.set_size_inches(12,14)

#   # 最多25張
#   if num>25:num=25

#   # 一張一張畫
#   for i in range(0,num):

#     # 建立子圖形5*5(五行五列)
#     ax=plt.subplot(5,5,i+1)

#     # 畫出子圖形
#     ax.imshow(images[idx],cmap='binary')

#     # 標題和label
#     title="label=" +str(labels[idx])

#     # 如果有傳入預測結果也顯示
#     if len(prediction)>0:
#       title+=",predict="+str(prediction[idx])

#     # 設定子圖形的標題大小
#     ax.set_title(title,fontsize=10)

#     # 設定不顯示刻度
#     ax.set_xticks([]);ax.set_yticks([])  
#     idx+=1
#   plt.show()  
  
# plot_images_labels_prediction(fix_train_images,train_labels,[],0,10)
# plot_images_labels_prediction(rotated[30000:],train_labels,[],0,10)
# plot_images_labels_prediction(rotated[60000:],train_labels,[],0,10)
# plot_images_labels_prediction(rotated[90000:],train_labels,[],0,10)
# plot_images_labels_prediction(rotated[120000:],train_labels,[],0,10)
# plot_images_labels_prediction(rotated[150000:],train_labels,[],0,10)
# plot_images_labels_prediction(rotated[180000:],train_labels,[],0,10)

# from keras import models
# from keras import layers
# import tensorflow as tf

 
# network = models.Sequential()
# network.add(layers.BatchNormalization())
# network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# network.add(layers.BatchNormalization())
# network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# network.add(layers.BatchNormalization())
# network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# network.add(layers.Dense(10, activation='softmax'))
# network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])

# from keras import models, optimizers
# from keras.layers import *
# network = models.Sequential()

# network.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
# network.add(BatchNormalization())
# network.add(Conv2D(filters=48, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
# network.add(BatchNormalization())
# network.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
# network.add(BatchNormalization())
# network.add(Conv2D(filters=80, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
# network.add(BatchNormalization())
# network.add(Conv2D(filters=96, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
# network.add(BatchNormalization())
# network.add(Conv2D(filters=112, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
# network.add(BatchNormalization())
# network.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
# network.add(BatchNormalization())
# network.add(Conv2D(filters=144, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
# network.add(BatchNormalization())
# network.add(Conv2D(filters=160, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
# network.add(BatchNormalization())
# network.add(Conv2D(filters=176, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
# network.add(BatchNormalization())


# network.add(Dropout(0.25))
# network.add(Flatten())
# network.add(Dense(128,activation='relu'))
# network.add(Dropout(0.25))
# network.add(Dense(10,activation='softmax'))

# adam = optimizers.Adam(lr=0.001, decay=0.98, amsgrad=False)
# network.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy']
#              )
# network.summary()


from keras import models, optimizers
from keras.layers import *
model = models.Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(28,28,1), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))
adam = optimizers.Adam(lr=0.001, decay=0.98, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
             )
model.summary()

result = model.fit(
    fix_train_images,
    fix_train_labels,
    epochs=30,
    batch_size=128,
    validation_data=(fix_test_images, fix_test_labels))

test_loss, test_acc = model.evaluate(fix_test_images, fix_test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)

history_dict = result.history
 
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
 
import matplotlib.pyplot as plt
plt.plot(epochs, loss_values, 'g', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.legend()
 
plt.show()

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
 
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0.98, 1)
plt.savefig("3*3kernel_accuracy_history.jpg") 
plt.show()

with open("/content/drive/MyDrive/Colab Notebooks/mnist/test.csv", newline='') as csvfile:
  rows = csv.reader(csvfile)
  data = np.array([row for row in rows])
real_test = data[1:].reshape((28000, 28 , 28)).astype('float32') / 255
prediction = model.predict(real_test)
header = ["ImageId", "Label"]

with open('/content/drive/MyDrive/Colab Notebooks/mnist/submit002.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    # Use writerows() not writerow()
    for i, p in enumerate(prediction):
      writer.writerow([i+1, np.argmax(p)])

