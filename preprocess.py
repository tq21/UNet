import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_generator():
    seed = 123
    train_path = os.path.join('/content/train')

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    mask_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        classes=['image'],
        target_size=(128, 128),
        batch_size=20,
        class_mode=None,
        color_mode='grayscale',
        seed=seed
    )

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=['label'],
        target_size=(128, 128),
        batch_size=20,
        class_mode=None,
        color_mode='grayscale',
        seed=seed
    )

    train_generator = zip(train_generator, mask_generator)
    return train_generator
