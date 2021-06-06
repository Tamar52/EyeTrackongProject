import os

import tensorflow as tf
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, concatenate
from keras.models import Model
from keras.layers import Layer
from EyeTrackongProject.eye_tracking_features import EyeTrackingFeatures
import numpy as np

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

    def train_model(self):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.dev

        dataset_path = "/cvgl/group/GazeCapture/gazecapture"

        print("{} dataset: {}".format(args.data, dataset_path))

        # train parameters
        n_epoch = args.max_epoch
        batch_size = args.batch_size
        patience = args.patience

        # image parameter TODO: EXPORT TO MAIN!!!
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
        sgd = SGD(lr=1e-1, decay=5e-4, momentum=9e-1, nesterov=True)
        adam = Adam(lr=1e-3)

        # compile model
        model.compile(optimizer=adam, loss='mse')

        # data
        # todo: parameters not hardocoded

        # debug
        # x, y = load_batch([l[0:batch_size] for l in train_data], img_ch, img_cols, img_rows)
        # x, y = load_batch_from_names(train_names[0:batch_size], dataset_path, img_ch, img_cols, img_rows)

        model.fit_generator(
            generator=generator_train_data(train_names, dataset_path, batch_size, img_ch, img_cols, img_rows),
            steps_per_epoch=(len(train_names)) / batch_size,
            epochs=n_epoch,
            verbose=1,
            validation_data=generator_val_data(val_names, dataset_path, batch_size, img_ch, img_cols, img_rows),
            validation_steps=(len(val_names)) / batch_size,
            callbacks=[EarlyStopping(patience=patience),
                       ModelCheckpoint("weights_big/weights.{epoch:03d}-{val_loss:.5f}.hdf5",
                                       save_best_only=True)
                       ]
        )

    def test_model(self, args):

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.dev

        names_path = ""
        print("Names to test: {}".format(names_path))

        dataset_path = ""
        print("Dataset: {}".format(names_path))

        weights_path = ""
        print("Weights: {}".format(weights_path))

        # image parameter
        img_cols = 64
        img_rows = 64
        img_ch = 3

        # test parameter
        batch_size = 64
        chunk_size = 500

        # model
        model = self._get_eye_tracker_model(img_ch, img_cols, img_rows)

        # model summary
        model.summary()

        # weights
        print("Loading weights...")
        model.load_weights(weights_path)

        # data
        test_names = load_data_names(names_path)

        # limit amount of testing data
        # test_names = test_names[:1000]

        # results
        err_x = []
        err_y = []

        print("Loading testing data...")
        for it in list(range(0, len(test_names), chunk_size)):

            x, y = load_batch_from_names_fixed(test_names[it:it + chunk_size],  dataset_path, img_ch, img_cols, img_rows)
            # x, y = load_batch_from_names(test_names[it:it + chunk_size], dataset_path, img_ch, img_cols, img_rows)
            predictions = model.predict(x=x, batch_size=batch_size, verbose=1)

            # print and analyze predictions
            for i, prediction in enumerate(predictions):
                print("PR: {} {}".format(prediction[0], prediction[1]))
                print("GT: {} {} \n".format(y[i][0], y[i][1]))

                err_x.append(abs(prediction[0] - y[i][0]))
                err_y.append(abs(prediction[1] - y[i][1]))

        # mean absolute error
        mae_x = np.mean(err_x)
        mae_y = np.mean(err_y)

        # standard deviation
        std_x = np.std(err_x)
        std_y = np.std(err_y)

        # final results
        print("MAE: {} {} ( samples)".format(mae_x, mae_y))
        print("STD: {} {} ( samples)".format(std_x, std_y))