from preprocess import *
from model import *


def main(steps_per_epoch=100, epochs=10):
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model = unet()
    train_generator = create_generator()

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[model_checkpoint])


if __name__ == "__main__":
    main()

