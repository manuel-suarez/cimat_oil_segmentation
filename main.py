import os
import logging
import argparse
import pytorch_lightning as pl

from dataset import CimatOilSpillDataset
from model import CimatOilModel
from utils import save_figure, test_model
from pytorch_lightning.loggers import CSVLogger

