import os
import os.path

import torch
import torch.nn as nn
import wandb

import models.simple_architecture.data_utils as data_utils
import models.simple_architecture.train_utils as train_utils
from config_loader import load_config
from models.simple_architecture.simplemodel.model import SimpleRNN

LEARNING_RATE = 0.001
NUM_EPOCHS = 100000
NUM_WORKERS = 10
N_LAYERS = 1
BATCH_SIZE = 500
MODEL_INPUT_SIZE = 100
MODEL_OUTPUT_SIZE = 4
MODEL_HIDDEN_DIM = 20


def simplemodel_train(logger, use_backup=False):
    config = load_config()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        logger.error("Cuda is unavailable")

    seq, angles = data_utils.get_embedded_data(config)
    train_data, test_data = data_utils.get_dataset(seq, angles)
    train_dataloader, val_dataloader = data_utils.get_dataloaders(train_data, test_data, BATCH_SIZE)

    model = SimpleRNN(MODEL_INPUT_SIZE, MODEL_OUTPUT_SIZE, MODEL_HIDDEN_DIM, N_LAYERS)
    start_epoch, model = train_utils.try_load_model_backup(model, use_backup, logger, config)
    model.to(device)

    train_utils.initialize_wandb(model, config, N_LAYERS, BATCH_SIZE, 'simple-model-angles')

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_utils.train_model(train_dataloader, val_dataloader, model, loss, optimizer, NUM_EPOCHS, logger, device,
                            config, train_utils.angles_metrics_logger,
                            start_epoch=start_epoch, model_backup_path=config["PATH_TO_SIMPLEMODEL_BACKUP"],
                            num_epoch_before_backup=config["NUM_EPOCH_BEFORE_BACKUP"])
    train_utils.write_training_epoch(config, 0)
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
