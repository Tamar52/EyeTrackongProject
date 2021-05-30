import tensorflow as tf
import os
import pandas as pd
import numpy as np
from typing import Dict
from eye_tracking_features import EyeTrackingFeatures


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
    def __init__(self, path_to_prepared_data: str):

        self.path_to_prepared_data = path_to_prepared_data

    def load_images(self, metadata: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        metadata[EyeTrackingFeatures.LEFT_EYE_FRAME.value] = tf.keras.preprocessing.image.load_img(
            metadata[EyeTrackingFeatures.LEFT_EYE_FRAME_PATH.value].decode("utf-8") , target_size=(128, 128))
        metadata[EyeTrackingFeatures.RIGHT_EYE_FRAME.value] = tf.keras.preprocessing.image.load_img(
            metadata[EyeTrackingFeatures.RIGHT_EYE_FRAME_PATH.value].decode("utf-8") , target_size=(128, 128))
        metadata[EyeTrackingFeatures.FACE_GRID_FRAME.value] = tf.keras.preprocessing.image.load_img(
            metadata[EyeTrackingFeatures.FACE_GRID_FRAME_PATH.value].decode("utf-8") , target_size=(128, 128))
        return metadata

    def check(self):
        return

    def preprocess_data(self):
        meta_data = pd.read_csv(os.path.join(self.path_to_prepared_data, 'metadata.csv'))
        meta_data.fillna(value='', inplace=True)
        meta_dict = meta_data.to_dict('list')
        dataset = tf.data.Dataset.from_tensor_slices(meta_dict)
        # dataset = dataset.map(map_func=self.load_images())

        for item in dataset.as_numpy_iterator():
            item = dataset.map(map_func=self.load_images(item))

if __name__ == "__main__":
    eye_tracker_data = EyeTrackerData(r'C:\Users\Tamar\Desktop\hw\project\prepared_data')
    eye_tracker_data.preprocess_data()
    print('DONE')