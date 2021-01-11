import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import *

class ConvBlock(tf.keras.Model):
    def __init__(self, num_filters, kernel_size=(3, 3)):
        super(ConvBlock, self).__init__()

        self.conv = Conv2D(num_filters, kernel_size)
        self.bn = BatchNormalization()
        self.relu = Activation("relu")
        self.pooling = MaxPool2D((2, 2))

    def call(self, x, pool=True):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        if pool == True:
            x = self.pooling(x)

        return x

def build_model(input_shape):
    inputs = Input(input_shape)

    x = ConvBlock(32)(inputs)
    x = ConvBlock(64)(x)
    x = ConvBlock(128)(x, pool=False)

    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation="softmax")(x)

    model = tf.keras.models.Model(inputs, x)
    return model

if __name__ == "__main__":
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(f"Train: {x_train.shape}")
    print(f"Test: {x_test.shape}")
    x_train = x_train/255.0
    x_test = x_test/255.0

    model = build_model((32, 32, 3))
    model.summary()

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
    model.fit(x_train, y_train, epochs=10, batch_size=128)
    model.evaluate(x_test, y_test)
