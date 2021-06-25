import argparse
from eye_tracker_model import EyeTrackerModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action='store_true', help='train flag')
    parser.add_argument('-eval', action='store_true', help='evaluate flag')
    parser.add_argument('-data', type=str, default='small', help='which dataset, small or big')
    parser.add_argument('-max_epoch', '--max_epoch', type=int, default=100, help='max number of iterations (default 100)')
    parser.add_argument('-batch_size', type=int, default=2, help='batch size (default 50)')
    parser.add_argument('-patience', type=int, default=15, help='early stopping patience (default 10)')
    parser.add_argument('-dev', type=str, default="-1", help='what cpu or gpu (recommended) use to train the model')
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
                                         path_to_model=r'C:\Users\Tamar\PycharmProjects\EyeTrackingProject\EyeTrackongProject\weights_big\weights.001-245.97839.hdf5')
