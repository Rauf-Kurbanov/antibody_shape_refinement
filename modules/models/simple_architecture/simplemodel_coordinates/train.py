import os
import os.path

import torch
import torch.nn as nn
import wandb

import models.simple_architecture.data_utils as data_utils
import models.simple_architecture.train_utils as train_utils
from config_loader import load_config
from models.simple_architecture.model import SimpleRNN
from metrics.metric import distance_between_atoms, angles_between_atoms

MODEL_NAME = 'simple-coordinates'


class DistanceLoss(nn.Module):
    def __init__(self, on_cpu):
        super().__init__()
        self.mse_loss_function = nn.MSELoss()
        self.on_cpu = on_cpu

    def forward(self, pred, target):
        distances_pred = distance_between_atoms(pred, self.on_cpu)
        distances_target = distance_between_atoms(target, self.on_cpu)
        z = self.mse_loss_function(distances_pred, distances_target)
        return z


class AnglesLoss(nn.Module):
    def __init__(self, on_cpu):
        super().__init__()
        self.mse_loss_function = nn.MSELoss()
        self.on_cpu = on_cpu

    def forward(self, pred, target, lengths):
        angles_pred = angles_between_atoms(pred, lengths, self.on_cpu)
        angles_target = angles_between_atoms(target, lengths, self.on_cpu)
        z = self.mse_loss_function(angles_pred, angles_target)
        return z


class ComplexLoss(nn.Module):
    def __init__(self, on_cpu):
        super().__init__()
        self.mse_loss_function = nn.MSELoss()
        self.distance_loss_function = DistanceLoss(on_cpu)
        self.angles_loss_function = AnglesLoss(on_cpu)

    def forward(self, pred, target, lengths):
        mse_loss = self.mse_loss_function(pred, target)
        distance_loss = self.distance_loss_function(pred, target)
        angles_loss = self.angles_loss_function(pred, target, lengths)
        z = mse_loss + distance_loss + angles_loss
        # todo add distance between ends loss
        return z


def parse_parameters(args):
    # todo add default params or throw exception
    model_input_size = args['model_input_size']
    model_output_size = args['model_output_size']
    model_hidden_dim = args['model_hidden_dim']
    learning_rate = args['learning_rate']
    n_layers = args['n_layers']
    batch_size = args['batch_size']
    epochs = args['epochs']
    return {
        "input_size": model_input_size,
        "output_size": model_output_size,
        "hidden_dim": model_hidden_dim,
        "learning_rate": learning_rate,
        "n_layers": n_layers,
        "batch_size": batch_size,
        "epochs": epochs,
    }


def simplemodel_coord_train(logger, args, use_backup=False, debug=False):
    params = parse_parameters(args)
    config = load_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        logger.error("Cuda is unavailable")

    seq, coord = data_utils.get_embedded_data_coordinates(config)
    train_data, test_data = data_utils.get_dataset_coordinates(seq, coord)
    train_dataloader, val_dataloader = data_utils.get_dataloaders(train_data, test_data, params['batch_size'])

    model = SimpleRNN(params['input_size'], params['output_size'], params['hidden_dim'], params['n_layers'])
    start_epoch, model = train_utils.try_load_model_backup(model, MODEL_NAME, use_backup, logger, config)
    model.to(device)

    if not debug:
        train_utils.initialize_wandb(model, config, params['n_layers'], params['batch_size'],
                                     'simple-model-coordinates', params['hidden_dim'])

    loss = ComplexLoss(on_cpu=debug)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    scheduler = None
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.002, step_size_up=100,
    #                                               cycle_momentum=False)

    train_utils.train_model(train_dataloader, val_dataloader, model, MODEL_NAME, loss, optimizer, params['epochs'],
                            logger,
                            device,
                            config,
                            train_utils.coordinates_metrics_logger,
                            scheduler=scheduler,
                            start_epoch=start_epoch, model_backup_path=config["PATH_TO_SIMPLEMODEL_COORD_BACKUP"],
                            num_epoch_before_backup=config["NUM_EPOCH_BEFORE_BACKUP"],
                            debug=debug)
    train_utils.write_training_epoch(config, 0, MODEL_NAME, logger)
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model-coordinates.pt'))
