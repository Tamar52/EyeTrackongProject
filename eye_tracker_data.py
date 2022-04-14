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
        dataset[EyeTrackingFeatures.LEFT_EYE_FRAME.value] = tf.image.resize(tf.io.decode_jpeg(
            tf.io.read_file(dataset[EyeTrackingFeatures.LEFT_EYE_FRAME_PATH.value]), channels=3), size=(224, 224))
        dataset[EyeTrackingFeatures.RIGHT_EYE_FRAME.value] = tf.image.resize(tf.io.decode_jpeg(
            tf.io.read_file(dataset[EyeTrackingFeatures.RIGHT_EYE_FRAME_PATH.value]), channels=3), size=(224, 224))
        dataset[EyeTrackingFeatures.FACE_FRAME.value] = tf.image.resize(tf.io.decode_jpeg(
            tf.io.read_file(dataset[EyeTrackingFeatures.FACE_FRAME_PATH.value]), channels=3), size=(224, 224))
        dataset[EyeTrackingFeatures.FACE_GRID.value] = tf.numpy_function(self._make_grid, [
            dataset[EyeTrackingFeatures.LABEL_FACE_GRID.value]], Tout = tf.float32)
        return dataset

    def preprocess_data(self, set_mode: str) -> tf.data.Dataset:
        meta_data = pd.read_csv(os.path.join(self._path_to_prepared_data, 'metadata.csv'))
        if set_mode == 'all':
            meta_data_panda = meta_data
        else:
            meta_data_panda = meta_data.loc[meta_data[EyeTrackingFeatures.DATASET.value] == set_mode]
            meta_data_panda.fillna(value='', inplace=True)
        meta_dict = meta_data_panda.to_dict('list')
        #meta_data_panda_Iphone = meta_data_panda[meta_data_panda[EyeTrackingFeatures.DEVICE_NAME.value].str.contains('iPad')]
        #meta_data_panda_Iphone.fillna(value='', inplace=True)
        #meta_dict = meta_data_panda_Iphone.to_dict('list')
        # meta_data_panda.fillna(value='', inplace=True)
        # meta_dict = meta_data_panda.to_dict('list')

        # for x in meta_dict[EyeTrackingFeatures.LABEL_FACE_GRID.value]:
        #     try:
        #         eval(x.replace("[ ", "[").replace("  ", " ").replace(" ", ","))
        #     except Exception:
        #         loli = 3
        meta_dict[EyeTrackingFeatures.LABEL_FACE_GRID.value] = [eval(x) for x in meta_dict[EyeTrackingFeatures.LABEL_FACE_GRID.value]]
        dataset = tf.data.Dataset.from_tensor_slices(meta_dict)
        dataset = dataset.map(map_func=self._load_images)

        return dataset

    def get_panda_data(self, set_mode: str) -> pd.DataFrame:
        meta_data = pd.read_csv(os.path.join(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/original_meta_data', 'metadata.csv'))
        if set_mode == 'all':
            return meta_data
        meta_data_panda = meta_data.loc[meta_data[EyeTrackingFeatures.DATASET.value] == set_mode]
        meta_data_panda.fillna(value='', inplace=True)



if __name__ == "__main__":
    eye_tracker_data = EyeTrackerData(r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/prepared_data')
    dataset = eye_tracker_data.preprocess_data('train')
    import cv2
    for sample in dataset.as_numpy_iterator():
        cv2.imshow("left_eye_frame", sample["left_eye_frame"].astype(np.uint8))
        cv2.imshow("right_eye_frame", sample["right_eye_frame"].astype(np.uint8))
        cv2.imshow("face_frame", sample["face_frame"].astype(np.uint8))
        cv2.imshow("face_grid", cv2.resize(sample["face_grid"].reshape((25, 25)), dsize=None, fx=10, fy=10))
        cv2.waitKey(0)
    print('DONE')