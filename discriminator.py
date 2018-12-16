#!/usr/bin/python3
import argparse
import os

from encoder.sc2_dask_dataset import starcraft_dataset

BATCH_SIZE = 2048
NUM_EPOCHS = 20

if os.name == 'nt':
    DATASET_PATH = os.path.join("B:\\", "documents", "sc2_datasets", "wcs_montreal.zarr")
    VALIDATION_PATH = os.path.join("B:\\", "documents", "sc2_datasets", "wcs_global.zarr")
    OUT_PATH = os.path.join("B:", "documents", "sc2_discriminator-splitplayer.keras")
else:
    DATASET_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_datasets", "wcs_montreal.zarr")
    VALIDATION_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_datasets", "wcs_global.zarr")
    OUT_PATH = os.path.join("/media", "sf_B_DRIVE", "documents", "sc2_trained_model_shuffled.keras")


def discriminate(dataset_path, validation_path=None, out_path=None, pretrain_path=None):
    print("Loading training set %s..." % dataset_path)
    use_validation = validation_path is not None
    train_set = starcraft_dataset(
        dataset_path,
        train_percentage=1.0 if use_validation else 0.9,
        batch_size=BATCH_SIZE
    )

    if validation_path is not None:
        print("Loading validation set %s..." % validation_path)
        validation_set = starcraft_dataset(validation_path, batch_size=BATCH_SIZE)
    else:
        validation_set = starcraft_dataset(
            dataset_path,
            train_percentage=0.9,
            val_split=True,
            batch_size=BATCH_SIZE)

    model = build_model(train_set, training=True)
    model.summary()
    if pretrain_path is not None:
        print("Using pretrained weights from %s..." % pretrain_path)
        model.load_weights(pretrain_path)

    try:
        model.fit(
            x=train_set.all()[0],
            y=train_set.all()[1],
            sample_weight=train_set.all()[2],
            validation_data=validation_set.all(),
            shuffle='batch',
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS
        )
    except KeyboardInterrupt:
        pass
    if out_path is not None:
        print("Saving model to %s..." % out_path)
        model.save_weights(out_path)


if __name__ == '__main__':
    # some imports down here to prevent child processes from importing tensorflow
    import dask
    import keras as k
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    from encoder.sc2_model import build_model

    dask.config.set(scheduler="threads")
    tf.logging.set_verbosity(tf.logging.WARN)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    parser = argparse.ArgumentParser()
    parser.add_argument("--training-set", default=DATASET_PATH, type=str)
    parser.add_argument("--out-path", default=None, type=str)
    parser.add_argument("--in-path", default=None, type=str)
    parser.add_argument("--validation-set", default=None, type=str)
    args = parser.parse_args()

    discriminate(args.training_set, args.validation_set, args.out_path, args.in_path)
