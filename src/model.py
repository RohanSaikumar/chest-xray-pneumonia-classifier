# Model architecture
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121

def build_model():
    base_model = DenseNet121(
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model
