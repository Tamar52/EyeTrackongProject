import os
from argparse import Namespace
from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.engine.keras_tensor import KerasTensor

from EyeTrackongProject.eye_tracker_data import EyeTrackerData
from EyeTrackongProject.eye_tracking_features import EyeTrackingFeatures
import numpy as np
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import  EarlyStopping, ModelCheckpoint, TensorBoard

'''
tensorflow (keras) model for the eye_tracker.

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''


class ScaledSigmoid(Layer):
    def __init__(self, alpha, beta, **kwargs):
        self.alpha = alpha
        self.beta = beta
        super(ScaledSigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaledSigmoid, self).build(input_shape)

    def call(self, x, mask=None):
        return self.alpha / (1 + np.exp(-x / self.beta))

    @staticmethod
    def get_output_shape_for(input_shape):
        return input_shape


class EyeTrackerModel(tf.keras.Model):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self, img_ch, img_cols, img_rows):
        super(EyeTrackerModel, self).__init__()
        self._l_eye_input = Input(shape=(img_rows,img_cols, img_ch), name=EyeTrackingFeatures.LEFT_EYE_FRAME.value) # Input layout is BHWC
        self._r_eye_input = Input(shape=(img_rows, img_cols, img_ch), name=EyeTrackingFeatures.RIGHT_EYE_FRAME.value)  # Input layout is BHWC
        self._face_input = Input(shape=(img_rows, img_cols, img_ch), name=EyeTrackingFeatures.FACE_FRAME.value)  # Input layout is BHWC
        self._face_grid_input = Input(shape=(625,), name=EyeTrackingFeatures.FACE_GRID.value)  # Input layout is BHWC # TODO: get from args
        # TODO: add l2 regularization - READ about L2 regularization
        self._eye_model = tf.keras.Sequential(
            [Conv2D(96, (11, 11), activation=EyeTrackingFeatures.RELU.value),
             MaxPool2D(pool_size=(2, 2)),
             Conv2D(256, (5, 5), activation=EyeTrackingFeatures.RELU.value),
             MaxPool2D(pool_size=(2, 2)),
             Conv2D(384, (3, 3), activation=EyeTrackingFeatures.RELU.value),
             MaxPool2D(pool_size=(2, 2))])

        self._face_model = tf.keras.Sequential(
            [Conv2D(96, (11, 11), activation=EyeTrackingFeatures.RELU.value),
             MaxPool2D(pool_size=(2, 2)),
             Conv2D(256, (5, 5), activation=EyeTrackingFeatures.RELU.value),
             MaxPool2D(pool_size=(2, 2)),
             Conv2D(384, (3, 3), activation=EyeTrackingFeatures.RELU.value),
             MaxPool2D(pool_size=(2, 2))])

    def get_eye_tracker_model(self) -> tf.keras.models:
        # right eye model
        right_eye = self._r_eye_input / 127.5 - 1
        right_eye_features = self._eye_model(right_eye)

        # left eye model
        left_eye = self._l_eye_input / 127.5 - 1
        left_eye_features = self._eye_model(left_eye)

        # face model
        face_image = self._face_input / 127.5 - 1
        face_features = self._face_model(face_image)

        # dense layers for eyes
        eye_features = Concatenate()([left_eye_features, right_eye_features])
        eye_features = Flatten()(eye_features)
        fc_e1 = Dense(128, activation=EyeTrackingFeatures.RELU.value)(eye_features)

        # dense layers for face
        face_features = Flatten()(face_features)
        fc_f1 = Dense(128, activation=EyeTrackingFeatures.RELU.value)(face_features)
        fc_f2 = Dense(64, activation=EyeTrackingFeatures.RELU.value)(fc_f1)

        # dense layers for face grid
        fc_fg1 = Dense(256, activation=EyeTrackingFeatures.RELU.value)(self._face_grid_input)
        fc_fg2 = Dense(128, activation=EyeTrackingFeatures.RELU.value)(fc_fg1)

        # final dense layers
        h = Concatenate()([fc_e1, fc_f2, fc_fg2])
        fc1 = Dense(128, activation=EyeTrackingFeatures.RELU.value)(h)
        fc2 = Dense(2, activation=EyeTrackingFeatures.LINEAR.value)(fc1)

        # final model
        final_model = Model(
            inputs=[self._l_eye_input, self._r_eye_input, self._face_input, self._face_grid_input],
            outputs=[fc2])

        return final_model

    def _prepare_data_for_model(self, model:tf.keras.Model, data: tf.data.Dataset, split: str, args: Namespace):
        def _map_method(example: Dict[str, KerasTensor]) -> Tuple[Tuple[KerasTensor], Tuple[KerasTensor]]:
            in_features = tuple(example[name] for name in model.input_names)
            out_features = tf.stack([example[EyeTrackingFeatures.LABEL_DOT_X_CAM.value], example[EyeTrackingFeatures.LABEL_DOT_Y_CAM.value]], axis=0)
            return (in_features, out_features)

        data = data.map(_map_method)
        if split == "train":
            data = data.shuffle(buffer_size=100)# TODO: increase
        data = data.batch(batch_size=args.batch_size)
        data = data.prefetch(100)
        return data

    def train_model(self, args):

        # train parameters
        n_epoch = args.max_epoch
        batch_size = args.batch_size
        patience = args.patience

        # image parameter
        img_cols = 64
        img_rows = 64
        img_ch = 3

        # model
        model = self.get_eye_tracker_model()

        # model summary
        model.summary()

        # weights
        # print("Loading weights...",  end='')
        # weights_path = "weights/weights.003-4.05525.hdf5"
        # model.load_weights(weights_path)
        # print("Done.")

        # optimizer
        sgd = SGD(learning_rate=1e-1, decay=5e-4, momentum=9e-1, nesterov=True)
        adam = Adam(learning_rate=1e-3)

        # compile model
        model.compile(optimizer=adam, loss='mse')

        # data
        # todo: parameters not hardocoded

        # debug
        # x, y = load_batch([l[0:batch_size] for l in train_data], img_ch, img_cols, img_rows)
        # x, y = load_batch_from_names(train_names[0:batch_size], dataset_path, img_ch, img_cols, img_rows)
        eye_tracker_data = EyeTrackerData(path_to_prepared_data=r'C:\Users\Tamar\Desktop\hw\project\prepared_data')

        train_data = self._prepare_data_for_model(model=model, data=eye_tracker_data.preprocess_data(set_mode='train'), split='train', args=args)
        val_data = self._prepare_data_for_model(model=model, data=eye_tracker_data.preprocess_data(set_mode='test'), split='test', args=args)

        model.fit(train_data,
                  epochs=args.max_epoch,
                  callbacks=[EarlyStopping(patience=patience),
                             ModelCheckpoint("weights_big/weights.{epoch:03d}-{val_loss:.5f}.hdf5", # todo: add timestamp
                                             save_best_only=True,
                                             verbose=1),
                             TensorBoard("weights_big/logs", update_freq=10) # TODO: add timestamp
                             ],
                  validation_data=val_data)

    # def test_model(self, args):
    #
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.dev
    #
    #     names_path = ""
    #     print("Names to test: {}".format(names_path))
    #
    #     dataset_path = ""
    #     print("Dataset: {}".format(names_path))
    #
    #     weights_path = ""
    #     print("Weights: {}".format(weights_path))
    #
    #     # image parameter
    #     img_cols = 64
    #     img_rows = 64
    #     img_ch = 3
    #
    #     # test parameter
    #     batch_size = 64
    #     chunk_size = 500
    #
    #     # model
    #     model = self._get_eye_tracker_model(img_ch, img_cols, img_rows)
    #
    #     # model summary
    #     model.summary()
    #
    #     # weights
    #     print("Loading weights...")
    #     model.load_weights(weights_path)
    #
    #     # data
    #     test_names = load_data_names(names_path)
    #
    #     # limit amount of testing data
    #     # test_names = test_names[:1000]
    #
    #     # results
    #     err_x = []
    #     err_y = []
    #
    #     print("Loading testing data...")
    #     for it in list(range(0, len(test_names), chunk_size)):
    #
    #         x, y = load_batch_from_names_fixed(test_names[it:it + chunk_size],  dataset_path, img_ch, img_cols, img_rows)
    #         # x, y = load_batch_from_names(test_names[it:it + chunk_size], dataset_path, img_ch, img_cols, img_rows)
    #         predictions = model.predict(x=x, batch_size=batch_size, verbose=1)
    #
    #         # print and analyze predictions
    #         for i, prediction in enumerate(predictions):
    #             print("PR: {} {}".format(prediction[0], prediction[1]))
    #             print("GT: {} {} \n".format(y[i][0], y[i][1]))
    #
    #             err_x.append(abs(prediction[0] - y[i][0]))
    #             err_y.append(abs(prediction[1] - y[i][1]))
    #
    #     # mean absolute error
    #     mae_x = np.mean(err_x)
    #     mae_y = np.mean(err_y)
    #
    #     # standard deviation
    #     std_x = np.std(err_x)
    #     std_y = np.std(err_y)
    #
    #     # final results
    #     print("MAE: {} {} ( samples)".format(mae_x, mae_y))
    #     print("STD: {} {} ( samples)".format(std_x, std_y))