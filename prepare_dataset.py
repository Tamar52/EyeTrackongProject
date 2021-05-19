import shutil, os, argparse, json, re, sys
import numpy as np
import scipy.io as sio
from PIL import Image
import pandas as pd


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
        'recording_id': [],
        'frame_id': [],
        'label_dot_x_cam': [],
        'label_dot_y_cam': [],
        'label_face_grid': [],
        'device_name': [],
        'dataset': []
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
            meta['recording_id'] += [int(recording)]
            meta['frame_id'] += [frame]
            meta['label_dot_x_cam'] += [dot_info['XCam'][j]]
            meta['label_dot_y_cam'] += [dot_info['YCam'][j]]
            meta['label_face_grid'] += [face_grid_bbox[j, :]]
            meta['device_name'] += [info['DeviceName']]
            meta['dataset'] += [info['Dataset']]
    meta_panda = pd.DataFrame.from_dict(meta)
    meta_panda.to_csv(os.path.join(args.output_path, 'metadata.csv'))


    # Integrate
    meta['recording_id'] = np.stack(meta['recording_id'], axis=0).astype(np.int16)
    meta['frame_id'] = np.stack(meta['frame_id'], axis=0).astype(np.int32)
    meta['label_dot_x_cam'] = np.stack(meta['label_dot_x_cam'], axis=0)
    meta['label_dot_y_cam'] = np.stack(meta['label_dot_y_cam'], axis=0)
    meta['label_face_grid'] = np.stack(meta['label_face_grid'], axis=0).astype(np.uint8)

    # Load reference metadata
    print('Will compare to the reference GitHub dataset metadata.mat...')
    reference = sio.loadmat('./reference_metadata.mat', struct_as_record=False)
    reference['labelRecNum'] = reference['labelRecNum'].flatten()
    reference['frame_id'] = reference['frameIndex'].flatten()
    reference['label_dot_x_cam'] = reference['labelDotXCam'].flatten()
    reference['label_dot_y_cam'] = reference['labelDotYCam'].flatten()
    reference['labelTrain'] = reference['labelTrain'].flatten()
    reference['labelVal'] = reference['labelVal'].flatten()
    reference['labelTest'] = reference['labelTest'].flatten()

    # Find mapping
    m_key = np.array(['%05d_%05d' % (rec, frame) for rec, frame in zip(meta['recording_id'], meta['frame_id'])],
                    np.object)
    r_key = np.array(
        ['%05d_%05d' % (rec, frame) for rec, frame in zip(reference['labelRecNum'], reference['frame_id'])],
        np.object)
    m_index = {k: i for i, k in enumerate(m_key)}
    r_index = {k: i for i, k in enumerate(r_key)}
    m_t_o_r = np.zeros((len(m_key, )), int) - 1
    for i, k in enumerate(m_key):
        if k in r_index:
            m_t_o_r[i] = r_index[k]
        else:
            log_error('Did not find rec_frame %s from the new dataset in the reference dataset!' % k)
    r_t_o_m = np.zeros((len(r_key, )), int) - 1
    for i, k in enumerate(r_key):
        if k in m_index:
            r_t_o_m[i] = m_index[k]
        else:
            log_error(f'Did not find rec_frame {k} from the reference dataset in the new dataset!', critical=False)
            # break

    # Copy split from reference
    meta['labelTrain'] = np.zeros((len(meta['recording_id'], )), np.bool)
    meta['labelVal'] = np.ones((len(meta['recording_id'], )), np.bool)  # default choice
    meta['labelTest'] = np.zeros((len(meta['recording_id'], )), np.bool)

    valid_mapping_mask = m_t_o_r >= 0
    meta['labelTrain'][valid_mapping_mask] = reference['labelTrain'][m_t_o_r[valid_mapping_mask]]
    meta['labelVal'][valid_mapping_mask] = reference['labelVal'][m_t_o_r[valid_mapping_mask]]
    meta['labelTest'][valid_mapping_mask] = reference['labelTest'][m_t_o_r[valid_mapping_mask]]

    # Statistics
    n_missing = np.sum(r_t_o_m < 0)
    n_extra = np.sum(m_t_o_r < 0)
    total_match = len(m_key) == len(r_key) and np.all(np.equal(m_key, r_key))
    print('======================\n\tSummary\n======================')
    print('Total added %d frames from %d recordings.' % (len(meta['frame_id']), len(np.unique(meta['recording_id']))))
    if n_missing > 0:
        print(
            f'There are {n_missing} frames missing in the new dataset. This may affect the results. Check the log to see which files are missing.')
    else:
        print('There are no missing files.')
    if n_extra > 0:
        print(
            f'There are {n_extra} extra frames in the new dataset. This is generally ok as they were marked for validation split only.')
    else:
        print('There are no extra files that were not in the reference dataset.')
    if total_match:
        print('The new metadata.mat is an exact match to the reference from GitHub (including ordering)')

    # import pdb; pdb.set_trace()
    input("Press Enter to continue...")


if __name__ == "__main__":
    main()
    print('DONE')
