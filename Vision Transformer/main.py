import os
import torch
import pandas as pd
from sklearn import metrics
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import lung_finetune_flex
from utils import *
from dataset import LUNG
from PIL import Image



def main():
    fold = 1
    tag = '-vit_1_1_cv'
    # dataset = HER2ST(train=True, fold=fold)
    dataset = LUNG(train=True, fold=fold)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=3, shuffle=True)
    # model=STModel(n_genes=785,hidden_dim=1024,learning_rate=1e-5)
    model = lung_finetune_flex(n_layers=5, n_genes=1000, learning_rate=1e-4)
    model.phase = "reconstruction"
    trainer = pl.Trainer(gpus=1, max_epochs=200)
    trainer.fit(model, train_loader)

    
    trainer.save_checkpoint("model/lung_last_train_" + tag + '_' + str(fold) + ".ckpt")

if __name__ == "__main__":
    main()
