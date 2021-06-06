import shutil, os, argparse, json, re, sys
import numpy as np
from PIL import Image
import pandas as pd
from eye_tracking_features import EyeTrackingFeatures


"""
Prepares the GazeCapture dataset for use with the keras code. Crops images, compiles JSONs into metadata.mat

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

"""


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', required=True)
parser.add_argument('--output_path', required=True)
args = parser.parse_args()


def read_json(filename):
    if not os.path.isfile(filename):
        log_error(f'Warning: No such file {filename}')
        return None

    with open(filename) as f:
        try:
            data = json.load(f)
        except:
            data = None

    if data is None:
        log_error(f'Warning: Could not read file {filename}')
        return None

    return data


def prepare_path(path, clear=False):
    if not os.path.isdir(path):
        os.makedirs(path)
    if clear:
        files = os.listdir(path)
        for f in files:
            f_path = os.path.join(path, f)
            if os.path.isdir(f_path):
                shutil.rmtree(f_path)
            else:
                os.remove(f_path)

    return path


def log_error(msg, critical=False):
    print(msg)
    if critical:
        sys.exit(1)


def crop_image(img, bbox):
    bbox = np.array(bbox, int)

    a_src = np.maximum(bbox[:2], 0)
    b_src = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))

    a_dst = a_src - bbox[:2]
    b_dst = a_dst + (b_src - a_src)

    res = np.zeros((bbox[3], bbox[2], img.shape[2]), img.dtype)
    res[a_dst[1]:b_dst[1], a_dst[0]:b_dst[0], :] = img[a_src[1]:b_src[1], a_src[0]:b_src[0], :]

    return res


def main():
    if not os.path.isdir(args.dataset_path):
        raise RuntimeError('No such dataset folder %s!' % args.dataset_path)

    prepare_path(args.output_path)

    # list recordings
    recordings = os.listdir(args.dataset_path)
    recordings = np.array(recordings, np.object)
    recordings = recordings[[os.path.isdir(os.path.join(args.dataset_path, r)) for r in recordings]]
    recordings.sort()

    # Output structure
    meta = {
        EyeTrackingFeatures.RECORDING_ID.value: [],
        EyeTrackingFeatures.FRAME_ID.value: [],
        EyeTrackingFeatures.LABEL_DOT_X_CAM.value: [],
        EyeTrackingFeatures.LABEL_DOT_Y_CAM.value: [],
        EyeTrackingFeatures.LABEL_FACE_GRID.value: [],
        EyeTrackingFeatures.DEVICE_NAME.value: [],
        EyeTrackingFeatures.DATASET.value: [],
        EyeTrackingFeatures.FACE_FRAME_PATH.value: [],
        EyeTrackingFeatures.RIGHT_EYE_FRAME_PATH.value: [],
        EyeTrackingFeatures.LEFT_EYE_FRAME_PATH.value: [],

    }

    for i, recording in enumerate(recordings):
        print('[%d/%d] Processing recording %s (%.2f%%)' % (i, len(recordings), recording, i / len(recordings) * 100))
        rec_dir = os.path.join(args.dataset_path, recording)
        rec_dir_out = os.path.join(args.output_path, recording)

        # Read JSONs
        apple_face = read_json(os.path.join(rec_dir, 'appleFace.json'))
        if apple_face is None:
            continue
        apple_left_eye = read_json(os.path.join(rec_dir, 'appleLeftEye.json'))
        if apple_left_eye is None:
            continue
        apple_right_eye = read_json(os.path.join(rec_dir, 'appleRightEye.json'))
        if apple_right_eye is None:
            continue
        dot_info = read_json(os.path.join(rec_dir, 'dotInfo.json'))
        if dot_info is None:
            continue
        face_grid = read_json(os.path.join(rec_dir, 'faceGrid.json'))
        if face_grid is None:
            continue
        frames = read_json(os.path.join(rec_dir, 'frames.json'))
        if frames is None:
            continue
        info = read_json(os.path.join(rec_dir, 'info.json'))
        if info is None:
            continue
        # screen = readJson(os.path.join(recDir, 'screen.json'))
        # if screen is None:
        #     continue

        face_path = prepare_path(os.path.join(rec_dir_out, 'appleFace'))
        left_eye_path = prepare_path(os.path.join(rec_dir_out, 'appleLeftEye'))
        right_eye_path = prepare_path(os.path.join(rec_dir_out, 'appleRightEye'))

        # Preprocess
        all_valid = np.logical_and(np.logical_and(apple_face['IsValid'], apple_left_eye['IsValid']),
                                   np.logical_and(apple_right_eye['IsValid'], face_grid['IsValid']))
        if not np.any(all_valid):
            continue

        frames = np.array([int(re.match('(\d{5})\.jpg$', x).group(1)) for x in frames])

        bbox_from_json = lambda data: np.stack((data['X'], data['Y'], data['W'], data['H']), axis=1).astype(int)
        face_bbox = bbox_from_json(apple_face) + [-1, -1, 1, 1]  # for compatibility with matlab code
        left_eye_bbox = bbox_from_json(apple_left_eye) + [0, -1, 0, 0]
        right_eye_bbox = bbox_from_json(apple_right_eye) + [0, -1, 0, 0]
        left_eye_bbox[:, :2] += face_bbox[:, :2]  # relative to face
        right_eye_bbox[:, :2] += face_bbox[:, :2]
        face_grid_bbox = bbox_from_json(face_grid)

        for j, frame in enumerate(frames):
            # Can we use it?
            if not all_valid[j]:
                continue

            # Load image
            img_file = os.path.join(rec_dir, 'frames', '%05d.jpg' % frame)
            if not os.path.isfile(img_file):
                log_error(f'Warning: Could not read image file {img_file}')
                continue
            img = Image.open(img_file)
            if img is None:
                log_error(f'Warning: Could not read image file {img_file}')
                continue
            img = np.array(img.convert('RGB'))

            # Crop images
            img_face = crop_image(img, face_bbox[j, :])
            img_eye_left = crop_image(img, left_eye_bbox[j, :])
            img_eye_right = crop_image(img, right_eye_bbox[j, :])

            # Save images
            Image.fromarray(img_face).save(os.path.join(face_path, '%05d.jpg' % frame), quality=95)
            Image.fromarray(img_eye_left).save(os.path.join(left_eye_path, '%05d.jpg' % frame), quality=95)
            Image.fromarray(img_eye_right).save(os.path.join(right_eye_path, '%05d.jpg' % frame), quality=95)

            # Collect metadata
            meta[EyeTrackingFeatures.RECORDING_ID.value] += [str(recording)]
            meta[EyeTrackingFeatures.FRAME_ID.value] += [str(frame)]
            meta[EyeTrackingFeatures.LABEL_DOT_X_CAM.value] += [str(dot_info['XCam'][j])]
            meta[EyeTrackingFeatures.LABEL_DOT_Y_CAM.value] += [str(dot_info['YCam'][j])]
            meta[EyeTrackingFeatures.LABEL_FACE_GRID.value] += [str(face_grid_bbox[j, :])]
            meta[EyeTrackingFeatures.DEVICE_NAME.value] += [info['DeviceName']]
            meta[EyeTrackingFeatures.DATASET.value] += [info['Dataset']]
            meta[EyeTrackingFeatures.FACE_FRAME_PATH.value] += [os.path.join(face_path, '%05d.jpg' % frame)]
            meta[EyeTrackingFeatures.RIGHT_EYE_FRAME_PATH.value] += [os.path.join(right_eye_path, '%05d.jpg' % frame)]
            meta[EyeTrackingFeatures.LEFT_EYE_FRAME_PATH.value] += [os.path.join(left_eye_path, '%05d.jpg' % frame)]

    meta_panda = pd.DataFrame.from_dict(meta)
    meta_panda.to_csv(os.path.join(args.output_path, 'metadata.csv'))


if __name__ == "__main__":
    main()
    print('DONE')
