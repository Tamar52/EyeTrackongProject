import tensorflow as tf
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, concatenate
from keras.models import Model

from EyeTrackongProject.eye_tracking_features import EyeTrackingFeatures

'''
tensorflow (keras) model for the eye_tracker.

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''


class EyeTrackerModel(tf.nn):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self, img_ch, img_cols, img_rows):
        super(EyeTrackerModel, self).__init__()
        self._input = Input(shape=(img_ch, img_cols, img_rows))
        self._eye_model = tf.keras.Sequential(
            [Conv2D(96, (11, 11), activation=EyeTrackingFeatures.RELU.value)(self._input),
             MaxPool2D(pool_size=(2, 2)),
             Conv2D(256, (5, 5), activation=EyeTrackingFeatures.RELU.value),
             MaxPool2D(pool_size=(2, 2)),
             Conv2D(384, (3, 3), activation=EyeTrackingFeatures.RELU.value),
             MaxPool2D(pool_size=(2, 2))])

        self._face_model = tf.keras.Sequential(
            [Conv2D(96, (11, 11), activation=EyeTrackingFeatures.RELU.value)(self._input),
             MaxPool2D(pool_size=(2, 2)),
             Conv2D(256, (5, 5), activation=EyeTrackingFeatures.RELU.value),
             MaxPool2D(pool_size=(2, 2)),
             Conv2D(384, (3, 3), activation=EyeTrackingFeatures.RELU.value),
             MaxPool2D(pool_size=(2, 2))])

    def get_eye_tracker_model(self) -> tf.keras.models:
        # right eye model
        right_eye_net = self._eye_model(self._input)

        # left eye model
        left_eye_net = self._eye_model(self._input)

        # face model
        face_net = self._face_model(self._input)

        # face grid
        face_grid = Input(shape=(1, 25, 25))

        # dense layers for eyes
        e = concatenate([left_eye_net, right_eye_net])
        e = Flatten()(e)
        fc_e1 = Dense(128, activation=EyeTrackingFeatures.RELU.value)(e)

        # dense layers for face
        f = Flatten()(face_net)
        fc_f1 = Dense(128, activation=EyeTrackingFeatures.RELU.value)(f)
        fc_f2 = Dense(64, activation=EyeTrackingFeatures.RELU.value)(fc_f1)

        # dense layers for face grid
        fg = Flatten()(face_grid)
        fc_fg1 = Dense(256, activation=EyeTrackingFeatures.RELU.value)(fg)
        fc_fg2 = Dense(128, activation=EyeTrackingFeatures.RELU.value)(fc_fg1)

        # final dense layers
        h = concatenate([fc_e1, fc_f2, fc_fg2])
        fc1 = Dense(128, activation=EyeTrackingFeatures.RELU.value)(h)
        fc2 = Dense(2, activation=EyeTrackingFeatures.LINEAR.value)(fc1)

        # final model
        final_model = Model(
            inputs=[self._input, self._input, self._input, face_grid],
            outputs=[fc2])

        return final_model
