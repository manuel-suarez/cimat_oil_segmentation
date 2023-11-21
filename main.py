import os
import logging
import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from dataset import CimatOilSpillDataset
from model import CimatOilModel
from utils import save_figure, test_model
from pytorch_lightning.loggers import CSVLogger

def create_datasets(data_dir, classes):
    return (
        CimatOilSpillDataset(data_dir, "train", classes=classes),
        CimatOilSpillDataset(data_dir, "cross", classes=classes)
    )

def create_dataloaders(n_cpu, train_dataset, valid_dataset):
    return (
        DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=n_cpu),
        DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=n_cpu)
    )

def process(input_dir, output_dir, arch, encoder):
    logging.info("Begin process")
    logging.info(f"\tArchitecture: {arch}")
    logging.info(f"\tEncoder: {encoder}")
    logging.info(f"\tInput dir: {input_dir}")
    logging.info(f"\tOutput dir: {output_dir}")
    classes = CimatOilSpillDataset.CLASSES
    train_dataset, valid_dataset = create_datasets(input_dir, classes=classes)
    logging.info("1.- Dataset configuration")
    logging.info(f"\tTrain dataset size: {len(train_dataset)}")
    logging.info(f"\tValid dataset size: {len(valid_dataset)}")

    train_dataloader, valid_dataloader = create_dataloaders(os.cpu_count(), train_dataset, valid_dataset)

    figures_dir = f"{arch}_figures"
    results_dir = f"{arch}_results"
    logs_dir = f"{arch}_logs"

    # Samples
    save_figure(train_dataset, "Train", os.path.join(output_dir, figures_dir, "figure_01.png"))
    save_figure(valid_dataset, "Valid", os.path.join(output_dir, figures_dir, "figure_02.png"))

def main(arch, encoder, input_dir, output_dir):
    process(input_dir, output_dir, arch, encoder)

parser = argparse.ArgumentParser(
    prog='Oil spill cimat dataset segmentation',
    description='Segmentation on Cimat oil spill dataset',
    epilog='With a great power comes a great responsability'
)
parser.add_argument('arch')
parser.add_argument('input_dir')
parser.add_argument('output_dir')
args = parser.parse_args()
arch = args.arch
input_dir = args.input_dir
output_dir = args.output_dir
logging.basicConfig(filename=f"{arch}_app.log", filemode='w', format='%(asctime)s: %(name)s %(levelname)s - %(message)s', level=logging.INFO)

# redirect lightning logging to file
logger = logging.getLogger("lightning.pytorch")
logger.addHandler(logging.FileHandler("core.log"))

logging.info("Start!")
encoder = 'resnet34'
main(arch, encoder, input_dir, output_dir)
logging.info("Done!")