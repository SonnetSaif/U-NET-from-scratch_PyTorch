import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# first layers
input = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
input = tf.keras.layers.Lambda(lamda x : x / 255)(input)
conv1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(input)
conv1 = tf.keras.layers.Dropout(0.1)(conv1)
pooling1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)
