import os
import os.path

import torch
import torch.nn as nn
import wandb

import models.simple_architecture.data_utils as data_utils
import models.simple_architecture.train_utils as train_utils
from config_loader import load_config
from models.simple_architecture.simplemodel_coordinates.model import SimpleRNN, SimpleAutoregressiveRNN
from metrics.metric import distance_between_atoms, angles_between_atoms

LEARNING_RATE = 0.001
NUM_EPOCHS = 100000
NUM_WORKERS = 1
# NUM_WORKERS = 16
N_LAYERS = 3
BATCH_SIZE = 500
MODEL_INPUT_SIZE = 100
MODEL_OUTPUT_SIZE = 9
MODEL_HIDDEN_DIM = 30
MODEL_NAME = 'simple-coordinates'


class DistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss_function = nn.MSELoss()

    def forward(self, pred, target):
        distances_pred = distance_between_atoms(pred)
        distances_target = distance_between_atoms(target)
        z = self.mse_loss_function(distances_pred, distances_target)
        return z


class AnglesLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss_function = nn.MSELoss()

    def forward(self, pred, target):
        angles_pred = angles_between_atoms(pred)
        angles_target = angles_between_atoms(target)
        z = self.mse_loss_function(angles_pred, angles_target)
        return z


class ComplexLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss_function = nn.MSELoss()
        self.distance_loss_function = DistanceLoss()
        self.angles_loss_function = AnglesLoss()

    def forward(self, pred, target):
        mse_loss = self.mse_loss_function(pred, target)
        distance_loss = self.distance_loss_function(pred, target)
        angles_loss = self.angles_loss_function(pred, target)
        z = mse_loss + distance_loss + angles_loss
        # todo add distance between ends loss
        return z


def simplemodel_coord_train(logger, use_backup=False, autoregressive=False):
    config = load_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        logger.error("Cuda is unavailable")

    seq, coord = data_utils.get_embedded_data_coordinates(config)
    train_data, test_data, train_data_nopad, test_data_nopad = data_utils.get_dataset_coordinates(seq, coord)
    train_dataloader, val_dataloader = data_utils.get_dataloaders(train_data, test_data, BATCH_SIZE)

    model = SimpleAutoregressiveRNN(MODEL_INPUT_SIZE, MODEL_OUTPUT_SIZE, MODEL_HIDDEN_DIM,
                                    1) if autoregressive else SimpleRNN(
        MODEL_INPUT_SIZE, MODEL_OUTPUT_SIZE, MODEL_HIDDEN_DIM, N_LAYERS)
    start_epoch = 0
    # start_epoch, model = train_utils.try_load_model_backup(model, MODEL_NAME, use_backup, logger, config)
    model.to(device)

    # train_utils.initialize_wandb(model, config, N_LAYERS, BATCH_SIZE, 'simple-model-coordinates', MODEL_HIDDEN_DIM)

    loss = ComplexLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.002, step_size_up=100,
                                                  cycle_momentum=False)

    if autoregressive:
        train_utils.train_model_autoregressive(train_dataloader, val_dataloader, model, MODEL_NAME, loss, optimizer,
                                               scheduler,
                                               NUM_EPOCHS,
                                               logger,
                                               device,
                                               config,
                                               train_utils.coordinates_metrics_logger,
                                               start_epoch=start_epoch,
                                               model_backup_path=config["PATH_TO_SIMPLEMODEL_COORD_BACKUP"],
                                               num_epoch_before_backup=config["NUM_EPOCH_BEFORE_BACKUP"])
    else:
        train_utils.train_model(train_dataloader, val_dataloader, model, MODEL_NAME, loss, optimizer, scheduler,
                                NUM_EPOCHS,
                                logger,
                                device,
                                config,
                                train_utils.coordinates_metrics_logger,
                                start_epoch=start_epoch, model_backup_path=config["PATH_TO_SIMPLEMODEL_COORD_BACKUP"],
                                num_epoch_before_backup=config["NUM_EPOCH_BEFORE_BACKUP"])
    train_utils.write_training_epoch(config, 0, MODEL_NAME, logger)
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model-coordinates.pt'))
