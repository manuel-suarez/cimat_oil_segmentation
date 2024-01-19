import os
import argparse
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import CSVLogger
from dataset import CimatOilSpillDataset, create_datasets, create_dataloaders
from model import CimatOilSpillModel
from pprint import pprint

classes = CimatOilSpillDataset.CLASSES

def process(input_dir, output_dir, arch, encoder, train_dataset, cross_dataset, test_dataset, num_epochs):
    train_dataset, valid_dataset, test_dataset = create_datasets(input_dir, train_dataset, cross_dataset, test_dataset)
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(os.cpu_count(), train_dataset, valid_dataset,
                                                                         test_dataset)
    base_dir = "results"
    figures_dir = os.path.join(base_dir, f"{arch}_figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir, exist_ok=True)

    encoder = "resnet34"
    model = CimatOilSpillModel(arch, encoder=encoder, in_channels=3, out_classes=1)

    checkpoint_path = os.path.join("/home/manuelsuarez/projects/smpt-50/training/Unet_results/Unet_logs/default/version_0/checkpoints", "epoch=49-step=80799.ckpt")
    trainer = pl.Trainer(accelerator="cpu",
                         max_epochs=num_epochs,
                         default_root_dir='results',
                         resume_from_checkpoint=checkpoint_path)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)

    valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    pprint(valid_metrics)
    # run test dataset
    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    pprint(test_metrics)

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
        plt.close()

def test(arch, encoder, input_dir, output_dir, train_dataset, cross_dataset, test_dataset, num_epochs):
    process(input_dir, output_dir, arch, encoder, train_dataset, cross_dataset, test_dataset, num_epochs)

if __name__ == "__main__":
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
