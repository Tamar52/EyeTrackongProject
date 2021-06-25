import os
from argparse import Namespace
from datetime import datetime
from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.keras.layers import BatchNormalization

from EyeTrackongProject.eye_tracker_data import EyeTrackerData
from EyeTrackongProject.eye_tracking_features import EyeTrackingFeatures
import numpy as np
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

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
        self._l_eye_input = Input(shape=(img_rows, img_cols, img_ch),
                                  name=EyeTrackingFeatures.LEFT_EYE_FRAME.value)  # Input layout is BHWC
        self._r_eye_input = Input(shape=(img_rows, img_cols, img_ch),
                                  name=EyeTrackingFeatures.RIGHT_EYE_FRAME.value)  # Input layout is BHWC
        self._face_input = Input(shape=(img_rows, img_cols, img_ch),
                                 name=EyeTrackingFeatures.FACE_FRAME.value)  # Input layout is BHWC
        self._face_grid_input = Input(shape=(625,),
                                      name=EyeTrackingFeatures.FACE_GRID.value)  # Input layout is BHWC # TODO: get from args
        # TODO: add l2 regularization - READ about L2 regularization
        self._eye_model = tf.keras.Sequential(
            [Conv2D(96, (11, 11), activation=EyeTrackingFeatures.RELU.value),
             BatchNormalization(),
             MaxPool2D(pool_size=(2, 2)),
             Conv2D(256, (5, 5), activation=EyeTrackingFeatures.RELU.value),
             BatchNormalization(),
             MaxPool2D(pool_size=(2, 2)),
             Conv2D(384, (3, 3), activation=EyeTrackingFeatures.RELU.value),
             BatchNormalization(),
             Conv2D(64, (1, 1), activation=EyeTrackingFeatures.RELU.value),
             BatchNormalization()])

        self._face_model = tf.keras.Sequential(
            [Conv2D(96, (11, 11), activation=EyeTrackingFeatures.RELU.value),
             BatchNormalization(),
             MaxPool2D(pool_size=(2, 2)),
             Conv2D(256, (5, 5), activation=EyeTrackingFeatures.RELU.value),
             BatchNormalization(),
             MaxPool2D(pool_size=(2, 2)),
             Conv2D(384, (3, 3), activation=EyeTrackingFeatures.RELU.value),
             BatchNormalization(),
             Conv2D(64, (1, 1), activation=EyeTrackingFeatures.RELU.value),
             BatchNormalization()])

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

        # eye features
        eye_features = Concatenate()([left_eye_features, right_eye_features])
        eye_features = Flatten()(eye_features)
        fc_e1 = Dense(128, activation=EyeTrackingFeatures.RELU.value, kernel_regularizer='l2')(eye_features)
        fc_e1 = BatchNormalization()(fc_e1)

        # dense layers for face
        face_features = Flatten()(face_features)
        face_features = BatchNormalization()(face_features)
        fc_f1 = Dense(128, activation=EyeTrackingFeatures.RELU.value,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.1))(face_features)
        fc_f1 = BatchNormalization()(fc_f1)
        fc_f2 = Dense(64, activation=EyeTrackingFeatures.RELU.value,
                      kernel_regularizer=tf.keras.regularizers.l2(l=0.1))(fc_f1)
        fc_f2 = BatchNormalization()(fc_f2)

        # dense layers for face grid
        fc_fg1 = Dense(256, activation=EyeTrackingFeatures.RELU.value,
                       kernel_regularizer=tf.keras.regularizers.l2(l=0.1))(self._face_grid_input)
        fc_fg1 = BatchNormalization()(fc_fg1)
        fc_fg2 = Dense(128, activation=EyeTrackingFeatures.RELU.value,
                       kernel_regularizer=tf.keras.regularizers.l2(l=0.1))(fc_fg1)
        fc_fg2 = BatchNormalization()(fc_fg2)

        # final dense layers
        h = Concatenate()([fc_e1, fc_f2, fc_fg2])
        fc1 = Dense(128, activation=EyeTrackingFeatures.RELU.value, kernel_regularizer=tf.keras.regularizers.l2(l=0.1))(
            h)
        fc1 = BatchNormalization()(fc1)
        fc2 = Dense(2, kernel_regularizer=tf.keras.regularizers.l2(l=0.1))(fc1)

        # final model
        final_model = Model(
            inputs=[self._l_eye_input, self._r_eye_input, self._face_input, self._face_grid_input],
            outputs=[fc2])

        return final_model

    def _prepare_data_for_model(self, model: tf.keras.Model, data: tf.data.Dataset, split: str, args: Namespace):
        def _map_method(example: Dict[str, KerasTensor]) -> Tuple[Tuple[KerasTensor], Tuple[KerasTensor]]:
            in_features = tuple(example[name] for name in model.input_names)
            out_features = tf.stack([example[EyeTrackingFeatures.LABEL_DOT_X_CAM.value],
                                     example[EyeTrackingFeatures.LABEL_DOT_Y_CAM.value]], axis=0)
            return (in_features, out_features)

        data = data.map(_map_method)
        if split == "train":
            data = data.shuffle(buffer_size=100)  # TODO: increase
        data = data.batch(batch_size=args.batch_size)
        data = data.prefetch(100)
        return data

    def train_model(self, args):
        # train parameters
        patience = args.patience

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

        eye_tracker_data = EyeTrackerData(path_to_prepared_data=r'C:\Users\Tamar\Desktop\hw\project\prepared_data')

        train_data = self._prepare_data_for_model(model=model, data=eye_tracker_data.preprocess_data(set_mode='train'),
                                                  split='train', args=args)
        val_data = self._prepare_data_for_model(model=model, data=eye_tracker_data.preprocess_data(set_mode='val'),
                                                split='val', args=args)

        time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        model.fit(train_data,
                  epochs=args.max_epoch,
                  callbacks=[EarlyStopping(patience=patience),
                             ModelCheckpoint("weights_big/weights.{epoch:03d}-{val_loss:.5f}_" + time_stamp + ".hdf5",
                                             save_best_only=True,
                                             verbose=1),
                             TensorBoard(f"weights_big/logs_{time_stamp}", update_freq=10)
                             ],
                  validation_data=val_data)

    def evaluate_model(self, path_to_model: str, args):
        model_to_eval = tf.keras.models.load_model(path_to_model)
        model_to_eval.summary()
        eye_tracker_data = EyeTrackerData(path_to_prepared_data=r'C:\Users\Tamar\Desktop\hw\project\prepared_data')
        test_data = self._prepare_data_for_model(model=model_to_eval, data=eye_tracker_data.preprocess_data(set_mode='test'),
                                                  split='test', args=args)
        loss = model_to_eval.evaluate(test_data, verbose=2)
        print(f"loss:{loss}")
