# Training and fine-tuning logic
import tensorflow as tf

def train_frozen(model, train_gen, val_gen, epochs=6):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    return model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[lr_scheduler],
        verbose=2
    )


def fine_tune(model, base_model, train_gen, val_gen, epochs=8):
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )

    return model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        verbose=2
    )
