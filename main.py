import os
import torch
import logging
import argparse
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset import CimatOilSpillDataset
from model import CimatOilSpillModel
from utils import save_figure, test_model
from pprint import pprint
from pytorch_lightning.loggers import CSVLogger

def create_datasets(data_dir, train_dataset, cross_dataset, test_dataset):
    featuresPath = os.path.join(data_dir, 'features')
    labelsPath = os.path.join(data_dir, 'labels')
    featureExt = '.tiff'
    labelExt = '.pgm'
    dims = [224, 224, 3]
    featuresChannels = ['ORIGIN', 'ORIGIN', 'VAR']
    trainingSet = pd.read_csv(train_dataset)
    crossvalidSet = pd.read_csv(cross_dataset)
    testingSet = pd.read_csv(test_dataset)
    return (
        CimatOilSpillDataset(trainingSet["key"], featuresPath, labelsPath, featuresChannels, featureExt, labelExt, dims),
        CimatOilSpillDataset(crossvalidSet["key"], featuresPath, labelsPath, featuresChannels, featureExt, labelExt, dims),
        CimatOilSpillDataset(testingSet['key'], featuresPath, labelsPath, featuresChannels, featureExt, labelExt, dims)
    )

def create_dataloaders(n_cpu, train_dataset, valid_dataset, test_dataset):
    return (
        DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=n_cpu),
        DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=n_cpu),
        DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=n_cpu)
    )

def process(input_dir, output_dir, arch, encoder, train_dataset, cross_dataset, test_dataset, num_epochs):
    logging.info("Begin process")
    logging.info(f"\tArchitecture: {arch}")
    logging.info(f"\tEncoder: {encoder}")
    logging.info(f"\tInput dir: {input_dir}")
    logging.info(f"\tOutput dir: {output_dir}")
    classes = CimatOilSpillDataset.CLASSES
    train_dataset, valid_dataset, test_dataset = create_datasets(input_dir, train_dataset, cross_dataset, test_dataset)

    logging.info("1.- Dataset configuration")
    logging.info(f"\tTrain dataset size: {len(train_dataset)}")
    logging.info(f"\tValid dataset size: {len(valid_dataset)}")
    logging.info(f"\tTest dataset size: {len(test_dataset)}")

    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(os.cpu_count(), train_dataset, valid_dataset, test_dataset)

    base_dir = "results"
    figures_dir = os.path.join(base_dir, f"{arch}_figures")
    results_dir = os.path.join(base_dir, f"{arch}_results")
    logs_dir = os.path.join(base_dir, f"{arch}_logs")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    # Samples
    save_figure(train_dataset, "Train", os.path.join(figures_dir, "figure_01.png"))
    save_figure(valid_dataset, "Valid", os.path.join(figures_dir, "figure_02.png"))
    save_figure(test_dataset, "Test", os.path.join(figures_dir, "figure_03.png"))

    logging.info("2.- Model instantiation")
    encoder = "resnet34"
    model = CimatOilSpillModel(arch, encoder=encoder, in_channels=3, out_classes=1)

    logging.info("3.- Model training")
    logger = CSVLogger(logs_dir)
    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, logger=logger, default_root_dir='results')
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.save_checkpoint(f"{arch}_{encoder}_{num_epochs}epochs.ckpt")

    logging.info("4.- Validation and test metrics")
    # run validation dataset
    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)
    # run test dataset
    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    pprint(test_metrics)

    logging.info("5.- Result visualization")
    batch = next(iter(test_dataloader))
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_labels = logits.sigmoid()
    for index, (image, gt_label, pr_label) in enumerate(zip(batch["image"], batch["label"], pr_labels)):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(image.numpy().transpose(1, 2, 0)) # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_label.numpy().squeeze()) # only one class
        plt.title("Label")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_label.numpy().squeeze())
        plt.title("Prediction")
        plt.axis("off")

        plt.savefig(os.path.join(figures_dir, f"figure_0{index+3+1}.png"))

def main(arch, encoder, input_dir, output_dir, train_dataset, cross_dataset, test_dataset, num_epochs):
    process(input_dir, output_dir, arch, encoder, train_dataset, cross_dataset, test_dataset, num_epochs)

if __name__ == 'main':
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
    logging.basicConfig(filename=f"{arch}_app.log", filemode='w', format='%(asctime)s: %(name)s %(levelname)s - %(message)s', level=logging.INFO)

    # redirect lightning logging to file
    logger = logging.getLogger("lightning.pytorch")
    #logger.addHandler(logging.FileHandler("core.log"))

    logging.info("Start!")
    encoder = 'resnet34'
    main(arch, encoder, args.input_dir, args.output_dir, args.train_dataset, args.cross_dataset, args.test_dataset, int(args.num_epochs))
    logging.info("Done!")