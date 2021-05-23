import enum


class EyeTrackingFeatures(enum.Enum):
    """
    enum for holding the eye tracking features
    """
    RECORDING_ID = 'recording_id'
    FRAME_ID = 'frame_id'
    LABEL_DOT_X_CAM = 'label_dot_x_cam'
    LABEL_DOT_Y_CAM = 'label_dot_y_cam'
    LABEL_FACE_GRID = 'label_face_grid'
    DEVICE_NAME = 'device_name'
    DATASET = 'dataset'
    FACE_GRID_FRAME_PATH = 'face_grid_frame_path'
    RIGHT_EYE_FRAME_PATH = 'right_eye_frame_path'
    LEFT_EYE_FRAME_PATH = 'left_eye_frame_path'
    FACE_GRID_FRAME = 'face_grid_frame'
    RIGHT_EYE_FRAME = 'right_eye_frame'
    LEFT_EYE_FRAME = 'left_eye_frame'

    def __str__(self) -> str:
        return self.value
