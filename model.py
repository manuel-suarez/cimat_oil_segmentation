import torch
import logging
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT


class CimatOilSpillModel(pl.LightningModule):
    def __init__(self, arch, encoder, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder, in_channels=in_channels, classes=out_classes, **kwargs
        )
        self.classes = out_classes

        # preprocessing parameters for image
        params = smp.encoders.get_preprocessing_params(encoder)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # binary segmentation loss
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        logging.debug(f"Forward, image shape: {image.shape}")
        image = (image - self.mean) / self.std
        predicted = self.model(image)
        return predicted

    def shared_step(self, batch, stage):
        image = batch["image"]
        bs = image.shape[0]
        h, w = image.shape[2:]

        # Shape of the image should be (batch_size, num_channels, height, widt)
        logging.debug(f"Shared step, stage: {stage}, image shape: {image.shape}")
        assert image.ndim == 4
        assert image.shape == (bs, 3, h, w) # Multichannel image composition (depending on configuration)

        # Check image dimensions are divisible by 32 (dimensionality reduction)
        assert h % 32 == 0 and w % 32 == 0

        label = batch["label"]

        # Shape of the label should be (batch_size, 1, height, width) binary classification
        logging.debug(f"Shared step, stage: {stage}, label shape: {label.shape}")
        assert label.ndim == 4
        assert label.shape == (bs, self.classes, h, w)

        # Check that label values are between 0 and 1
        assert label.max() <= 1 and label.min() >= 0

        logits = self.forward(image)
        # Predicted label contains logits, for that reason loss function param 'from_logits' must be set to true
        loss = self.loss_fn(logits, label)

        # Metrics

        probs = logits.sigmoid()
        preds = (probs > 0.5).float()

        # IoU
        tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), label.long(), mode="binary")
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

    def shared_epoch_end(self, outputs, stage):
        # Metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # IoU per image
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # IoU per dataset
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou
        }
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        logging.debug(f"Training Step, batch idx: {batch_idx}")
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        logging.debug(f"Validation step, batch idx: {batch_idx}")
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)