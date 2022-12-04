import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

input = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
input = tf.keras.layers.Lambda(lamda x : x / 255)(input)

TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print("Resizing training image and masks")
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# first layer
conv1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(input)
conv1 = tf.keras.layers.Dropout(0.1)(conv1)
conv1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
pooling1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)

# contraction path
conv2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(pooling1)
conv2 = tf.keras.layers.Dropout(0.1)(conv2)
conv2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
pooling2 = tf.keras.layers.MaxPooling2D((2,2))(conv2)

conv3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(pooling2)
conv3 = tf.keras.layers.Dropout(0.1)(conv3)
conv3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
pooling3 = tf.keras.layers.MaxPooling2D((2,2))(conv3)

conv4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(pooling3)
conv4 = tf.keras.layers.Dropout(0.1)(conv4)
conv4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)
pooling4 = tf.keras.layers.MaxPooling2D((2,2))(conv4)

conv5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(pooling4)
conv5 = tf.keras.layers.Dropout(0.1)(conv5)
conv5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)

# expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv5)
u6 = tf.keras.layers.concatenate([u6, conv4])
conv6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
conv6 = tf.keras.layers.Dropout(0.1)(conv6)
conv6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv6)
u7 = tf.keras.layers.concatenate([u7, conv3])
conv7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
conv7 = tf.keras.layers.Dropout(0.1)(conv7)
conv7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(conv7)
u8 = tf.keras.layers.concatenate([u8, conv2])
conv8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
conv8 = tf.keras.layers.Dropout(0.1)(conv8)
conv8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv8)

u9 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(conv8)
u9 = tf.keras.layers.concatenate([u9, conv1], axis=3)
conv9 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
conv9 = tf.keras.layers.Dropout(0.1)(conv9)
conv9 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(conv9)

output = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[input], outputs=[output])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

check_point = tf.keras.callbacks.ModelCheckpoint('u-net_from_scratch.h5', verbose=1, save_best_only)

call_backs = [
    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]

output = model.fit(X, Y, validation_split=0.2, epochs=50, batch_size=32, callbacks=call_backs)
