import os
import os.path
import pickle

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config_loader import load_config
from metrics.metric import coordinate_metrics
from models.simplemodel.model import SimpleRNN

LEARNING_RATE = 0.001
NUM_EPOCHS = 100000
NUM_WORKERS = 16
N_LAYERS = 2
BATCH_SIZE = 500
MODEL_INPUT_SIZE = 100
MODEL_OUTPUT_SIZE = 9
MODEL_HIDDEN_DIM = 20


def pad_data(X_train, y_train, X_test, y_test, lengths_train, lengths_test, logger):
    X_train_padded = rnn_utils.pad_sequence(X_train, batch_first=True)
    y_train_padded = rnn_utils.pad_sequence(y_train, batch_first=True)
    train_dataset = [[X_train_padded[i], y_train_padded[i], lengths_train[i]] for i in range(len(X_train_padded))]

    X_test_padded = rnn_utils.pad_sequence(X_test, batch_first=True)
    y_test_padded = rnn_utils.pad_sequence(y_test, batch_first=True)
    test_dataset = [[X_test_padded[i], y_test_padded[i], lengths_test[i]] for i in range(len(X_test_padded))]

    return train_dataset, test_dataset


def get_dataset(seq, coordinates, logger):
    full_data = [[len(seq[x]), seq[x], coordinates[x.decode('ascii')]] for x in seq.keys()]
    test_data = full_data[:int(len(full_data) / 10)]
    train_data = full_data[int(len(full_data) / 10):]

    full_data.sort(key=lambda x: x[0], reverse=True)
    # TODO log

    train_data.sort(key=lambda x: x[0], reverse=True)
    lengths_train = [train_data[i][0] for i in range(len(train_data))]
    X_train = [train_data[i][1] for i in range(len(train_data))]
    y_train = [train_data[i][2] for i in range(len(train_data))]

    test_data.sort(key=lambda x: x[0], reverse=True)
    lengths_test = [test_data[i][0] for i in range(len(test_data))]
    X_test = [test_data[i][1] for i in range(len(test_data))]
    y_test = [test_data[i][2] for i in range(len(test_data))]

    train_dataset, test_dataset = pad_data(X_train, y_train, X_test, y_test, lengths_train, lengths_test, logger)

    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, batch_size):
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    return train_dataloader, val_dataloader


def train_model(train_dataloader, val_dataloader, model, loss, optimizer, num_epochs, logger, device, config,
                model_backup_path=None, start_epoch=0, num_epoch_before_backup=100):
    for epoch in range(start_epoch, num_epochs):
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')
        # TODO epoch logging
        # all_preds = []
        # all_lengths = []
        # all_targets = []

        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                model.train()
            else:
                dataloader = val_dataloader
                model.eval()

            running_loss = 0.

            for inputs, targets, lengths in tqdm(dataloader):
                inputs = inputs.to(device)

                # repad target sequences
                targets = rnn_utils.pack_padded_sequence(targets, lengths, batch_first=True)
                targets, _ = rnn_utils.pad_packed_sequence(targets, batch_first=True)
                targets = targets.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds, lengths, hiddens = model(inputs, lengths)

                    loss_value = loss(preds, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                    else:
                        # all_preds.append(preds)
                        # all_targets.append(targets)
                        # all_lengths.append(lengths)

                        metrics = coordinate_metrics(preds, targets, lengths, device)

                        wandb.log({"MAE batch": metrics['mae']})
                        wandb.log({"Distance deviation between ends": metrics['diff_ends_dist']})
                        wandb.log({"Distance deviation between neighbours": metrics['diff_neighbours_dist']})
                        wandb.log({"Percent distance deviation between ends": metrics['diff_ends_dist_p']})
                        wandb.log({"Percent distance deviation between neighbours": metrics['diff_neighbours_dist_p']})

                # statistics
                running_loss += loss_value.item()

            epoch_loss = running_loss / len(dataloader)

            logger.info(f'{phase} Loss: {epoch_loss}')
            if phase == 'train':
                wandb.log({"Train loss": epoch_loss})
            else:
                wandb.log({"Test loss": epoch_loss})

        if epoch % num_epoch_before_backup == 0 and model_backup_path:
            torch.save(model.state_dict(), model_backup_path)
            write_training_epoch(config, epoch)

    return model


def get_embedded_data(config):
    seq = {}
    coordinates = {}
    with open(config["PATH_TO_SEQ_EMBEDDED"], 'rb') as handle:
        seq = pickle.load(handle)
    with open(config["PATH_TO_COORD_EMBEDDED"], 'rb') as handle:
        coordinates = pickle.load(handle)
    return seq, coordinates


def write_training_epoch(config, epoch):
    with open(config["PATH_TO_FINISHED_TRAINING_SIMPLEMODEL_COORD"], 'w') as f:
        f.write(f"{epoch}")


def check_training_epoch(config):
    # if not os.path.isfile(config["PATH_TO_FINISHED_TRAINING_SIMPLEMODEL_COORD"]):
    #     return 0
    # with open(config["PATH_TO_FINISHED_TRAINING_SIMPLEMODEL_COORD"], 'r') as f:
    #     epoch = int(f.read())
    #     return epoch
    return 0


def try_load_unfinished_model(logger, config):
    try:
        state = torch.load(config["PATH_TO_SIMPLEMODEL_COORD_BACKUP"])
        return state
    except:
        logger.exception('Error loading unfinished simplemodel_coordinates')


def simplemodel_coord_train(logger):
    config = load_config()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        logger.error("Cuda is unavailable")

    seq, coord = get_embedded_data(config)

    train_data, test_data = get_dataset(seq, coord, logger)
    train_dataloader, val_dataloader = get_dataloaders(train_data, test_data, BATCH_SIZE)

    model = SimpleRNN(MODEL_INPUT_SIZE, MODEL_OUTPUT_SIZE, MODEL_HIDDEN_DIM, N_LAYERS)
    start_epoch = check_training_epoch(config)
    logger.info(f'Starting training from epoch {start_epoch}')
    if start_epoch > 0:
        state = try_load_unfinished_model(logger, config)
        if state:
            logger.info(f'Successfully loaded backup model')
            model.load_state_dict(state)
    model.to(device)

    wandb.init(project=config["PROJECT_NAME"],
               name=f"basic-model-coordinates n_layers={N_LAYERS} batch_size={BATCH_SIZE}")
    wandb.watch(model)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(train_dataloader, val_dataloader, model, loss, optimizer, NUM_EPOCHS, logger, device, config,
                start_epoch=start_epoch, model_backup_path=config["PATH_TO_SIMPLEMODEL_COORD_BACKUP"],
                num_epoch_before_backup=config["NUM_EPOCH_BEFORE_BACKUP"])
    write_training_epoch(config, 0)
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model-coordinates.pt'))
