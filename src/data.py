# Data loading and generators
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input

def create_generators(
    csv_path,
    train_dir,
    test_dir,
    batch_size=32,
    img_size=(224, 224)
):
    df = pd.read_csv(csv_path)

    train_df = df[df["Dataset_type"] == "TRAIN"].reset_index(drop=True)
    test_df  = df[df["Dataset_type"] == "TEST"].reset_index(drop=True)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        samplewise_center=True,
        samplewise_std_normalization=True,
        preprocessing_function=preprocess_input,
        validation_split=0.1
    )

    train_gen = datagen.flow_from_dataframe(
        train_df,
        directory=train_dir,
        x_col="X_ray_image_name",
        y_col="Label",
        target_size=img_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=True,
        subset="training"
    )

    val_gen = datagen.flow_from_dataframe(
        train_df,
        directory=train_dir,
        x_col="X_ray_image_name",
        y_col="Label",
        target_size=img_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False,
        subset="validation"
    )

    test_gen = datagen.flow_from_dataframe(
        test_df,
        directory=test_dir,
        x_col="X_ray_image_name",
        y_col="Label",
        target_size=img_size,
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False
    )

    return train_gen, val_gen, test_gen
