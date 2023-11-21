import os
import torch
import numpy as np
import logging

from skimage.io import imread

class CimatOilSpillDataset(torch.utils.data.Dataset):
    # List of classes (binary segmentation)
    CLASSES = ['oil', 'not oil']

    def __init__(self,
                 keys,              # List of file names
                 featuresPath,      # Path of feature files
                 labelsPath,        # Path of label files
                 featuresChannels,  # Feature channels specification
                 featureExt,        # File extension of feature files
                 labelExt,          # File extension of label files
                 dims
                 ):
        self.keys = keys
        self.dims = dims
        self.labelsPath = labelsPath
        self.featuresPath = featuresPath
        self.featuresChannels = featuresChannels
        self.featureExt = featureExt
        self.labelExt = labelExt

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # Key index (filename)
        key = self.keys[index]
        # Open feature channels and build multichannel image
        image = np.zeros(self.dims, dtype=np.float32)
        for j, feature in enumerate(self.featuresChannels):
            feature_path = os.path.join(self.featuresPath, feature, key + self.featureExt)
            feature_image = imread(feature_path, as_gray=True).astype(np.float32)

            image[...,j] = feature_image
        # Convert to pytorch format: HWC -> CHW
        image = np.moveaxis(image, -1, 0)
        # Open label
        label_path = os.path.join(self.labelsPath, key + self.labelExt)
        label = imread(label_path, as_gray=True).astype(np.float32)/255.0
        # Return data
        return dict(image=image, label=label)