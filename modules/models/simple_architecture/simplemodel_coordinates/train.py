import os
import os.path

import torch
import torch.nn as nn
import wandb

import models.simple_architecture.data_utils as data_utils
import models.simple_architecture.train_utils as train_utils
from config_loader import load_config
from models.simple_architecture.simplemodel_coordinates.model import SimpleRNN

LEARNING_RATE = 0.001
NUM_EPOCHS = 100000
NUM_WORKERS = 16
N_LAYERS = 2
BATCH_SIZE = 500
MODEL_INPUT_SIZE = 100
MODEL_OUTPUT_SIZE = 9
MODEL_HIDDEN_DIM = 40
MODEL_NAME = 'simple-coordinates'


def distances_between_atoms(loop):
    def roll(x):
        return torch.cat((x[:, -1:, :], x[:, :-1, :]), 1)
    loop = loop.view(loop.shape[0], -1, 3)
    rolled_loop = roll(loop)
    cropped_loop = loop[:, :-1]
    cropped_rolled_loop = rolled_loop[:, :-1]
    return torch.sqrt(torch.sum((cropped_loop - cropped_rolled_loop) ** 2, dim=-1))


class DistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss_function = nn.MSELoss()

    def forward(self, pred, target):
        distances_pred = distances_between_atoms(pred)
        distances_target = distances_between_atoms(target)
        z = self.mse_loss_function(distances_pred, distances_target)
        return z


class ComplexLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss_function = nn.MSELoss()
        self.distance_loss_function = DistanceLoss()

    def forward(self, pred, target):
        mse_loss = self.mse_loss_function(pred, target)
        distance_loss = self.distance_loss_function(pred, target)
        z = mse_loss + distance_loss
        return z


def simplemodel_coord_train(logger, use_backup=False):
    config = load_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        logger.error("Cuda is unavailable")

    seq, coord = data_utils.get_embedded_data_coordinates(config)
    train_data, test_data = data_utils.get_dataset_coordinates(seq, coord)
    train_dataloader, val_dataloader = data_utils.get_dataloaders(train_data, test_data, BATCH_SIZE)

    model = SimpleRNN(MODEL_INPUT_SIZE, MODEL_OUTPUT_SIZE, MODEL_HIDDEN_DIM, N_LAYERS)
    start_epoch, model = train_utils.try_load_model_backup(model, MODEL_NAME, use_backup, logger, config)
    model.to(device)

    train_utils.initialize_wandb(model, config, N_LAYERS, BATCH_SIZE, 'simple-model-coordinates')

    loss = ComplexLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_utils.train_model(train_dataloader, val_dataloader, model, MODEL_NAME, loss, optimizer, NUM_EPOCHS, logger,
                            device,
                            config,
                            train_utils.coordinates_metrics_logger,
                            start_epoch=start_epoch, model_backup_path=config["PATH_TO_SIMPLEMODEL_COORD_BACKUP"],
                            num_epoch_before_backup=config["NUM_EPOCH_BEFORE_BACKUP"])
    train_utils.write_training_epoch(config, 0, MODEL_NAME, logger)
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model-coordinates.pt'))
