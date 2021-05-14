import scipy.io as sio
from PIL import Image
import os
import os.path
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


MEAN_PATH = './'


def load_metadata(filename, silent=False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print(f'\tReading metadata from {filename}...')
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print(f'\tFailed to read the meta file "{filename}"!')
        return None
    return metadata


class SubtractMean(object):
    """Normalize a tensor image with mean.
    """

    def __init__(self, mean_img):
        self.mean_img = transforms.ToTensor()(mean_img / 255)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        return tensor.sub(self.mean_img)


class ITrackerData(data.Dataset):
    def __init__(self, dataPath, split='train', imSize=(224, 224), gridSize=(25, 25)):

        self.dataPath = dataPath
        self.imSize = imSize
        self.gridSize = gridSize

        print('Loading iTracker dataset...')
        metaFile = os.path.join(dataPath, 'metadata.mat')
        # metaFile = 'metadata.mat'
        if metaFile is None or not os.path.isfile(metaFile):
            raise RuntimeError('There is no such file %s! Provide a valid dataset path.' % metaFile)
        self.metadata = load_metadata(metaFile)
        if self.metadata is None:
            raise RuntimeError('Could not read metadata file %s! Provide a valid dataset path.' % metaFile)

        self.faceMean = load_metadata(os.path.join(MEAN_PATH, 'mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = load_metadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'))['image_mean']
        self.eyeRightMean = load_metadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'))['image_mean']

        self.transformFace = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.faceMean),
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeLeftMean),
        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeRightMean),
        ])

        if split == 'test':
            mask = self.metadata['labelTest']
        elif split == 'val':
            mask = self.metadata['labelVal']
        else:
            mask = self.metadata['labelTrain']

        self.indices = np.argwhere(mask)[:, 0]
        print('Loaded iTracker dataset split "%s" with %d records...' % (split, len(self.indices)))

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            # im = Image.new("RGB", self.imSize, "white")

        return im

    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen, ], np.float32)

        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2])
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3])
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid

    def __getitem__(self, index):
        index = self.indices[index]

        imFacePath = os.path.join(self.dataPath, '%05d/appleFace/%05d.jpg' % (
        self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
        imEyeLPath = os.path.join(self.dataPath, '%05d/appleLeftEye/%05d.jpg' % (
        self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))
        imEyeRPath = os.path.join(self.dataPath, '%05d/appleRightEye/%05d.jpg' % (
        self.metadata['labelRecNum'][index], self.metadata['frameIndex'][index]))

        imFace = self.loadImage(imFacePath)
        imEyeL = self.loadImage(imEyeLPath)
        imEyeR = self.loadImage(imEyeRPath)

        imFace = self.transformFace(imFace)
        imEyeL = self.transformEyeL(imEyeL)
        imEyeR = self.transformEyeR(imEyeR)

        gaze = np.array([self.metadata['labelDotXCam'][index], self.metadata['labelDotYCam'][index]], np.float32)

        faceGrid = self.makeGrid(self.metadata['labelFaceGrid'][index, :])

        # to tensor
        row = torch.LongTensor([int(index)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)

        return row, imFace, imEyeL, imEyeR, faceGrid, gaze

    def __len__(self):
        return len(self.indices)
