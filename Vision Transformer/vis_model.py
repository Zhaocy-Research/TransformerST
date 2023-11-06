import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from transformer import ViT
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

class lung_finetune_flex(pl.LightningModule):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.1, n_pos=128):
        super().__init__()
        self.learning_rate = learning_rate
        patch_dim = 3 * patch_size * patch_size
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.computation_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.dim=1024
        # self.x_embed = nn.Embedding(n_pos, dim)
        # self.y_embed = nn.Embedding(n_pos, dim)
        self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2 * dim, dropout=dropout, emb_dropout=dropout)
        self.phase = "reconstruction"  # Set initial phase
        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, patches,centers):
        # _, centers, _ = patches.size()
        centers_x = self.x_embed(centers[:, :, 0])
        centers_y = self.y_embed(centers[:, :, 1])

        patches = self.patch_embedding(patches)
        x = patches + centers_x + centers_y
        h = self.vit(x)
        # print(h.shape,'shape')
        if self.phase == "reconstruction":
            gene_recon = self.gene_head(h)
            return gene_recon
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")

    def one_hot_encode(self,labels, num_classes):
        return torch.eye(num_classes)[labels]
    def check_for_invalid_values(self,tensor, name):
        if torch.isnan(tensor).any():
            print(f"{name} contains NaN values!")
        if torch.isinf(tensor).any():
            print(f"{name} contains Inf values!")


    def training_step(self, batch, batch_idx):
        patch, centers,target_gene = batch
        if self.phase == "reconstruction":
            pred_gene = self(patch,centers)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            loss = loss_gene
            self.log('train_loss_recon', loss)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")

        return loss

    def validation_step(self, batch, batch_idx):
        patch, centers,target_gene = batch  # assuming masks are the segmentation ground truth

        if self.phase == "reconstruction":
            pred_gene = self(patch,centers)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            loss = loss_gene
            self.log('eval_loss_recon', loss)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")

        return loss

    def test_step(self, batch, batch_idx):
        patch, centers, target_gene = batch  # assuming masks are the segmentation ground truth
        if self.phase == "reconstruction":
            pred_gene = self(patch,centers)
            loss_gene = F.mse_loss(pred_gene.view_as(target_gene), target_gene)
            loss = loss_gene
            self.log('test_loss_recon', loss)
        else:
            raise ValueError("Invalid phase. Choose between 'reconstruction' and 'segmentation'.")
        return loss
    def reconstruction_parameters(self):
        return list(self.gene_head.parameters())

    def configure_optimizers(self):
        if self.phase == "reconstruction":
            optimizer = torch.optim.Adam(self.reconstruction_parameters(), lr=1e-3)
        return optimizer

if __name__ == '__main__':
    a = torch.rand(1,4000,3*112*112)
    p = torch.ones(1,4000,2).long()
    model = lung_finetune_flex()
    x = model(a,p)
