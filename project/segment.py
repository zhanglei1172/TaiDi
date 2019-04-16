#!/usr/bin/python
# -*- coding: utf-8 -*-
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
# from sqlalchemy.sql.operators import custom_op
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from config import *
from preprocessing import *

GAUSSIAN_NOISE = 0.1
# UPSAMPLE_MODE = 'SIMPLE'
UPSAMPLE_MODE = 'DECONV'
# downsampling inside the network
NET_SCALING = None
# downsampling in preprocessing
IMG_SCALING = (1, 1)
# EPSILON = 1e-5

BATCH_SIZE = 1
MIN_BOUND = -1000.0
MAX_BOUND = 400.0

data_gen_args = dict(
    rotation_range=90.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    brightness_range=(0.5, 1.0),
    validation_split=0.2
)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, image_datagen, list_IDs, labels, batch_size=32, dim=(IMG_HEIGHT, IMG_WIDTH),
                 n_channels=IMG_CHANNELS,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.image_datagen = image_datagen
        self.batch_size = batch_size
        self.labels = labels
        self.dcm_series = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dcm_series) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # X = np.clip(np.expand_dims(get_pixels_hu(
        #     [dicom.read_file(self.dcm_series[k]) for k in indexes]), -1), -100, 100) / 100.0
        # Find list of IDs
        # list_IDs_temp = [self.dcm_series[k] for k in indexes]
        # X = (np.clip(np.expand_dims(get_pixels_hu(
        #     [dicom.read_file(self.dcm_series[k]) for k in indexes]), -1), -400, 300))/70
        X = np.clip((np.expand_dims(get_pixels_hu(
            [dicom.read_file(self.dcm_series[k]) for k in indexes]), -1) - MIN_BOUND) / (MAX_BOUND - MIN_BOUND), 0.,
                    1.) - 0.25

        y = np.expand_dims(np.array([read_mask(self.labels[k]) for k in indexes])/255, -1)
        # Generate data
        # X, y = self.__data_generation(list_IDs_temp)

        return self.image_datagen.flow(X, y, self.batch_size).next()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dcm_series))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # def __data_generation(self, list_IDs_temp):
    #     'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    #     # Initialization
    #     X = np.empty((self.batch_size, *self.dim, self.n_channels))
    #     y = np.empty((self.batch_size), dtype=int)
    #
    #     # Generate data
    #     for i, ID in enumerate(list_IDs_temp):
    #         # Store sample
    #         X[i,] = np.load('data/' + ID + '.npy')
    #
    #         # Store class
    #         y[i] = self.labels[ID]


if __name__ == '__main__':
    train = 1
    image_datagen = ImageDataGenerator(**data_gen_args)

    df = process_data("CT影像/*/*/*.dcm")
    df_train, df_val = train_test_split(df, test_size=0.15, random_state=0)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    if train:
        # df_train, df_val = train_test_split(df, test_size=0.15, random_state=0)
        # df_train = df_train.reset_index(drop=True)
        # df_val = df_val.reset_index(drop=True)
        training_generator = DataGenerator(image_datagen, df_train['path_dcm'], df_train['path_mask'],
                                           batch_size=BATCH_SIZE)
        validation_generator = DataGenerator(image_datagen, df_val['path_dcm'], df_val['path_mask'],
                                             batch_size=BATCH_SIZE)
        model = bulid_model()
        # earlystopper = EarlyStopping(patience=5, verbose=1)
        checkpointer = ModelCheckpoint('model-taidi.h5', verbose=1, save_best_only=False)
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            # use_multiprocessing=True,
                            workers=1,
                            epochs=50,
                            callbacks=[
                                # earlystopper,
                                checkpointer])
    else:
        model = load_model('./model-taidi.h5', custom_objects={'dice_coeff_loss': dice_coeff_loss,
                                                               'iou_coef': iou_coef})

        training_generator = DataGenerator(image_datagen, df_train['path_dcm'], df_train['path_mask'],
                                           batch_size=BATCH_SIZE, shuffle=False)
        validation_generator = DataGenerator(image_datagen, df_val['path_dcm'], df_val['path_mask'],
                                             batch_size=BATCH_SIZE, shuffle=False)
        x = model.predict_generator(validation_generator)
