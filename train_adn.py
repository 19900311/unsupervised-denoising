import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import lightning as L

from torchmetrics.image.fid import FrechetInceptionDistance

from adn import ADN
from radon_adn import RadonADN
from components import Discriminator

class TrainADN(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.radon:
            self.generator = RadonADN()
        else:
            self.generator = ADN()
        self.discriminator_artifact = Discriminator()
        self.discriminator_no_artifact = Discriminator()
        self.criterion = nn.BCEWithLogitsLoss()

        self.fid_train = FrechetInceptionDistance(normalize=True)
        self.fid_val = FrechetInceptionDistance(normalize=True)

        self.automatic_optimization = False

    def self_consistency_loss(self, y, y_denoised):
        return F.l1_loss(y_denoised, y)

    def artifact_consistency_loss(self, x, x_denoised, y, y_noisy):
        return F.l1_loss((x - x_denoised), (y_noisy - y))

    def reconstruction_loss(self, x, x_reconstructed, y, y_reconstructed):
        return F.l1_loss(x_reconstructed, x) + F.l1_loss(y_reconstructed, y)

    def forward(self, x, y):
        return self.generator(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch

        optimizer_G, optimizer_DA, optimizer_DNA = self.optimizers()
        x_denoised, x_reconstructed, y_noisy, y_denoised, y_reconstructed = self.generator(x, y)

        real_labels = torch.ones(x.size()[0], 1, 62, 62, device=self.device)
        fake_labels = torch.zeros(x.size()[0], 1, 62, 62, device=self.device)

        # Optimize Artifact Discriminator

        real_loss = self.criterion(
            self.discriminator_artifact(x),
            real_labels,
        )

        fake_loss = self.criterion(
            self.discriminator_artifact(y_noisy.detach()),
            fake_labels,
        )

        da_loss = real_loss + fake_loss
        optimizer_DA.zero_grad()
        self.manual_backward(da_loss)
        optimizer_DA.step()

        # Optimize No Artifact Discriminator

        real_loss = self.criterion(
            self.discriminator_no_artifact(y),
            real_labels,
        )

        fake_loss = self.criterion(
            self.discriminator_no_artifact(x_denoised.detach()),
            fake_labels,
        )

        dna_loss = real_loss + fake_loss
        optimizer_DNA.zero_grad()
        self.manual_backward(dna_loss)
        optimizer_DNA.step()

        # Optimize Generator

        adv_artifact = self.criterion(
            self.discriminator_artifact(y_noisy),
            real_labels,
        )

        adv_no_artifact = self.criterion(
            self.discriminator_no_artifact(x_denoised),
            real_labels,
        )

        loss_self = self.self_consistency_loss(y, y_denoised)
        loss_art = self.artifact_consistency_loss(x, x_denoised, y, y_noisy)
        loss_rec = self.reconstruction_loss(x, x_reconstructed, y, y_reconstructed)

        loss = self.hparams.w_adv * (adv_artifact + adv_no_artifact) + \
               self.hparams.w_loss * (loss_self + loss_art + loss_rec)
        optimizer_G.zero_grad()
        self.manual_backward(loss)
        optimizer_G.step()

        # Log Metrics

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        x, y = batch

        x_denoised = self.generator.simple_forward(x)

        y = y.repeat(1, 3, 1, 1)
        x_denoised = x_denoised.repeat(1, 3, 1, 1)

        y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
        x_denoised = (x_denoised - torch.min(x_denoised)) / (torch.max(x_denoised) - torch.min(x_denoised))

        self.fid_train.update(y, real=True)
        self.fid_train.update(x_denoised, real=False)

    def on_train_epoch_end(self):
        fid_score = self.fid_train.compute()
        self.log('train_fid', fid_score, on_epoch=True, prog_bar=True)
        self.fid_train.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x_denoised = self.generator.simple_forward(x)

        if batch_idx == 0:
            paired_images = torch.cat((x, x_denoised), dim=0)
            grid = vutils.make_grid(paired_images, nrow=4, normalize=True, scale_each=True)
            grid = transforms.ToPILImage()(grid)

            self.logger.experiment.log({"images_val": [wandb.Image(grid)]})

        y = y.repeat(1, 3, 1, 1)
        x_denoised = x_denoised.repeat(1, 3, 1, 1)

        y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
        x_denoised = (x_denoised - torch.min(x_denoised)) / (torch.max(x_denoised) - torch.min(x_denoised))

        self.fid_val.update(y, real=True)
        self.fid_val.update(x_denoised, real=False)

    def on_validation_epoch_end(self):
        fid_score = self.fid_val.compute()
        self.log('val_fid', fid_score, on_epoch=True, prog_bar=True)
        self.fid_val.reset()

    def configure_optimizers(self):
        lr = self.hparams.lr
        beta1 = self.hparams.beta1
        beta2 = self.hparams.beta2
        weight_decay = self.hparams.weight_decay

        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        optimizer_DA = torch.optim.Adam(self.discriminator_artifact.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        optimizer_DNA = torch.optim.Adam(self.discriminator_no_artifact.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
        return [optimizer_G, optimizer_DA, optimizer_DNA], []
