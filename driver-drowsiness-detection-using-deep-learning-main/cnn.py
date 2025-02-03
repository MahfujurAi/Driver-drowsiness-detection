from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import cv2
import os

train = ImageDataGenerator(rescale = 1/255)
val = ImageDataGenerator(rescale = 1/255)
test = ImageDataGenerator(rescale = 1/255)

train_data = train.flow_from_directory(
        '/kaggle/input/driver-drowsiness-dataset/Driver_Drowsiness_Dataset/train',
        target_size=(128, 128),
        batch_size=8,
        class_mode='binary'
    )
val_data = val.flow_from_directory(
        '/kaggle/input/driver-drowsiness-dataset/Driver_Drowsiness_Dataset/val',
        target_size=(128, 128),
        batch_size=8,
        class_mode='binary'
    )
test_data = test.flow_from_directory(
        '/kaggle/input/driver-drowsiness-dataset/Driver_Drowsiness_Dataset/test',
        target_size=(128, 128),
        batch_size=8,
        class_mode='binary'
    )


def create_custom_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', dilation_rate=2, input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', dilation_rate=2),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', dilation_rate=2),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    return model

input_shape = (128, 128, 3)
cnn_model = create_custom_cnn(input_shape)
cnn_model.summary()

cnn_model.compile(optimizer= RMSprop(learning_rate =0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model = cnn_model.fit(train_data, epochs=20 ,validation_data =val_data)
cnn_model.save('model.h5')
plt.plot(model.history['accuracy'],color='red',label='train')
plt.plot(model.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()
plt.plot(model.history['loss'],color='red',label='train')
plt.plot(model.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()
test_loss , test_acc =  cnn_model.evaluate(test_data , verbose=2)
print(test_acc)