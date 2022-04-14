import argparse
from eye_tracker_model import EyeTrackerModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action='store_true', help='train flag')
    parser.add_argument('-eval', action='store_true', help='evaluate flag')
    parser.add_argument('-predict', action='store_true', help='predict flag')
    parser.add_argument('-max_epoch', '--max_epoch', type=int, default=10000, help='max number of epochs')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-patience', type=int, default=20, help='early stopping patience')
    args = parser.parse_args()

    # image parameter
    img_cols = 224
    img_rows = 224
    img_ch = 3
    eye_tracker_model = EyeTrackerModel(img_ch=img_ch, img_cols=img_cols, img_rows=img_rows)

    # train
    if args.train:
        eye_tracker_model.train_model(args)

    # eval
    if args.eval:
        eye_tracker_model.evaluate_model(args=args,
                                         path_to_model=r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/first_model_normal_batch_32/weights.106-4.05546_2021-06-27_16-13-33.hdf5')

   # predict
    if args.predict:
        eye_tracker_model.predict_model(args=args,
                                         path_to_model=r'/mnt/2ef93ccf-c66e-4beb-95ba-24011e8fee18/TAMAR/filtered_data/normal_batch_32/weights.026-9.65809_2022-03-27_11-19-57.hdf5')
