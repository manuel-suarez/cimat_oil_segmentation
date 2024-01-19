import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from dataset import CimatOilSpillDataset, create_datasets, create_dataloaders
from model import CimatOilSpillModel

classes = CimatOilSpillDataset.CLASSES

def process(input_dir, output_dir, arch, encoder, train_dataset, cross_dataset, test_dataset, num_epochs):
    train_dataset, valid_dataset, test_dataset = create_datasets(input_dir, train_dataset, cross_dataset, test_dataset)
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(os.cpu_count(), train_dataset, valid_dataset,
                                                                         test_dataset)
    encoder = "resnet34"
    model = CimatOilSpillModel(arch, encoder=encoder, in_channels=3, out_classes=1)

    trainer = pl.Trainer(accelerator="cpu",
                         max_epochs=num_epochs,
                         default_root_dir='results',
                         resume_from_checkpoint=f"{arch}_{encoder}_{num_epochs}epochs.ckpt")
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)

def test(arch, encoder, input_dir, output_dir, train_dataset, cross_dataset, test_dataset, num_epochs):
    process(input_dir, output_dir, arch, encoder, train_dataset, cross_dataset, test_dataset, num_epochs)

parser = argparse.ArgumentParser(
    prog='Oil spill cimat dataset segmentation',
    description='Segmentation on Cimat oil spill dataset',
    epilog='With a great power comes a great responsability'
)
parser.add_argument('arch')
parser.add_argument('input_dir')
parser.add_argument('output_dir')
parser.add_argument('train_dataset')
parser.add_argument('cross_dataset')
parser.add_argument('test_dataset')
parser.add_argument('num_epochs')
args = parser.parse_args()
arch = args.arch
encoder = 'resnet34'
test(arch, encoder, args.input_dir, args.output_dir, args.train_dataset, args.cross_dataset, args.test_dataset, int(args.num_epochs))
