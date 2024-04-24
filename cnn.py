import os
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## 데이터셋 만들기
batch_size = 32
validation_split = 0.1
random_seed = 123
EPOCH = 50
learning_rate = 0.01


def basic_cnn(input_shape):
    cnn = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./255.0),
        tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=input_shape, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    return cnn

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


data_dir = "C:\\Users\\김민정\\Desktop\\초음파_modified CWT\\img\\RAW\\"

datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

train_ds = datagen.flow_from_directory(
    data_dir,
    target_size=(1500,800),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=random_seed
)

val_ds = datagen.flow_from_directory(
    data_dir,
    target_size=(1500,800),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=random_seed
)


model = basic_cnn(input_shape=(1500,800))


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(train_ds, epochs=EPOCH, validation_data=val_ds)