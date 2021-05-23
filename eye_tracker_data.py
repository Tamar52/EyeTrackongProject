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



def check():

    training_df: pd.DataFrame = pd.DataFrame(
        data={
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10),
            'feature3': np.random.rand(10),
            'target': np.random.randint(0, 3, 10)
        }
    )
    features = ['feature1', 'feature2', 'feature3']
    print(training_df)

    training_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(training_df[features].values, tf.float32),
                tf.cast(training_df['target'].values, tf.int32)
            )
        )
    )

    for features_tensor, target_tensor in training_dataset:
        print(f'features:{features_tensor} target:{target_tensor}')


class EyeTrackerData:
    def __init__(self, path_to_prepared_data: str):

        self.path_to_prepared_data = path_to_prepared_data

    def load_images(self, metadata: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        metadata[EyeTrackingFeatures.LEFT_EYE_FRAME.value] = tf.keras.preprocessing.image.load_img(
            metadata[EyeTrackingFeatures.LEFT_EYE_FRAME_PATH.value], target_size=(128, 128))
        metadata[EyeTrackingFeatures.RIGHT_EYE_FRAME.value] = tf.keras.preprocessing.image.load_img(
            metadata[EyeTrackingFeatures.RIGHT_EYE_FRAME_PATH.value], target_size=(128, 128))
        metadata[EyeTrackingFeatures.FACE_GRID_FRAME.value] = tf.keras.preprocessing.image.load_img(
            metadata[EyeTrackingFeatures.FACE_GRID_FRAME_PATH.value], target_size=(128, 128))
        return metadata

    def preprocess_data(self):
        meta_data = pd.read_csv(os.path.join(self.path_to_prepared_data, 'metadata.csv'))
        meta_data.fillna(value='', inplace=True)

        features = meta_data.keys()
        meta_dict = meta_data.to_dict('records')
        # training_dataset = tf.data.Dataset.from_tensor_slices(tf.cast(meta_data[features].values, tf.float32))
        dataset = tf.data.Dataset.from_tensor_slices(meta_dict)
        dataset = dataset.map(map_func=self.load_images(meta_data))

        for item in dataset.as_numpy_iterator():
            loli = 3


if __name__ == "__main__":
    eye_tracker_data = EyeTrackerData(r'C:\Users\Tamar\Desktop\hw\project\prepared_data')
    eye_tracker_data.preprocess_data()
    # check()
    print('DONE')