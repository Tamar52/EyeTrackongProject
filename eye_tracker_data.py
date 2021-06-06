import ast

import tensorflow as tf
import os
import pandas as pd
from eye_tracking_features import EyeTrackingFeatures
import numpy as np
"""
Data loader for the eyeTracker.

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}
"""


class EyeTrackerData:
    def __init__(self, path_to_prepared_data: str, img_size=(224,224), grid_size=(25, 25)):

        self._path_to_prepared_data = path_to_prepared_data
        self._img_size = img_size
        self._grid_size = grid_size


    def _make_grid(self, params):
        grid_len = self._grid_size[0] * self._grid_size[1]
        grid = np.zeros([grid_len, ], np.float32)

        inds_y = np.array([i // self._grid_size[0] for i in range(grid_len)])
        inds_x = np.array([i % self._grid_size[0] for i in range(grid_len)])
        cond_x = np.logical_and(inds_x >= params[0], inds_x < params[0] + params[2])
        cond_y = np.logical_and(inds_y >= params[1], inds_y < params[1] + params[3])
        cond = np.logical_and(cond_x, cond_y)

        grid[cond] = 1
        return grid

    def _load_images(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        dataset[EyeTrackingFeatures.LEFT_EYE_FRAME.value] = tf.io.decode_jpeg(
            tf.io.read_file(dataset[EyeTrackingFeatures.LEFT_EYE_FRAME_PATH.value]), channels=3)
        dataset[EyeTrackingFeatures.RIGHT_EYE_FRAME.value] = tf.io.decode_jpeg(
            tf.io.read_file(dataset[EyeTrackingFeatures.RIGHT_EYE_FRAME_PATH.value]), channels=3)
        dataset[EyeTrackingFeatures.FACE_FRAME.value] = tf.io.decode_jpeg(
            tf.io.read_file(dataset[EyeTrackingFeatures.FACE_FRAME_PATH.value]), channels=3)
        # dataset[EyeTrackingFeatures.FACE_GRID.value] = self._make_grid(
        #     dataset[EyeTrackingFeatures.LABEL_FACE_GRID.value])
        return dataset

    def preprocess_data(self):
        meta_data = pd.read_csv(os.path.join(self._path_to_prepared_data, 'metadata.csv'))
        meta_data.fillna(value='', inplace=True)
        meta_dict = meta_data.to_dict('list')
        dataset = tf.data.Dataset.from_tensor_slices(meta_dict)
        dataset = dataset.map(map_func=self._load_images)



if __name__ == "__main__":
    eye_tracker_data = EyeTrackerData(r'C:\Users\Tamar\Desktop\hw\project\prepared_data')
    eye_tracker_data.preprocess_data()
    print('DONE')