import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import time, math
from collections import OrderedDict

"""
Keras model for the eyeTracker.

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}
"""


class EyeTrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(EyeTrackerImageModel, self).__init__()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class FaceImageModel(nn.Module):

    def __init__(self):
        super(FaceImageModel, self).__init__()

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize=25):
        super(FaceGridModel, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class EyeTrackerModel(nn.Module):

    def __init__(self):
        super(EyeTrackerModel, self).__init__()


    def forward(self, faces, eyesLeft, eyesRight, faceGrids):
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)
        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)

        # Face net
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyes, xFace, xGrid), 1)
        x = self.fc(x)

        return x
