import os
import logging
import wandb
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modules.models.simplemodel.model import SimpleRNN
from modules.embedding import get_embeddings
from modules.embedding import get_angles
from modules.metrics.metric import angle_metrics

path_to_model = '/antibody-shape-refinement/data/embedding/pretrained_models/ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav'
path_to_seq = '/antibody-shape-refinement/data/antibodies/cdr_h3_seq'
path_to_angles = '/antibody-shape-refinement/data/antibodies/cdr_h3_angles'

num_workers = 10
n_layers = 2
batch_size = 500


def padd_data(X_train, y_train, X_test, y_test, lengths_train, lengths_test):
    X_train_padded = rnn_utils.pad_sequence(X_train, batch_first=True)
    y_train_padded = rnn_utils.pad_sequence(y_train, batch_first=True)
    train_dataset = [[X_train_padded[i], y_train_padded[i], lengths_train[i]] for i in range(len(X_train_padded))]

    X_test_padded = rnn_utils.pad_sequence(X_test, batch_first=True)
    y_test_padded = rnn_utils.pad_sequence(y_test, batch_first=True)
    test_dataset = [[X_test_padded[i], y_test_padded[i], lengths_test[i]] for i in range(len(X_test_padded))]

    return train_dataset, test_dataset


def get_dataset(seq, angles):
    full_data = [[len(seq[x]), seq[x], angles[x]] for x in seq.keys()]
    test_data = full_data[:int(len(full_data) / 10)]
    train_data = full_data[int(len(full_data) / 10):]

    full_data.sort(key=lambda x: x[0], reverse=True)
    # TODO log
    # print(len(test_data), len(train_data))

    train_data.sort(key=lambda x: x[0], reverse=True)
    lengths_train = [train_data[i][0] for i in range(len(train_data))]
    X_train = [train_data[i][1] for i in range(len(train_data))]
    y_train = [train_data[i][2] for i in range(len(train_data))]

    # print(lengths_train[0], X_train[0].shape, y_train[0].shape)

    test_data.sort(key=lambda x: x[0], reverse=True)
    lengths_test = [test_data[i][0] for i in range(len(test_data))]
    X_test = [test_data[i][1] for i in range(len(test_data))]
    y_test = [test_data[i][2] for i in range(len(test_data))]

    # print(lengths_test, X_test[0].shape, y_test[0].shape)

    train_dataset, test_dataset = padd_data(X_train, y_train, X_test, y_test, lengths_train, lengths_test)

    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, batch_size):
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

    return train_dataloader, val_dataloader


def train_model(train_dataloader, val_dataloader, model, loss, optimizer, num_epochs, logger, device):
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                model.train()  # Set model to training mode
            else:
                dataloader = val_dataloader
                model.eval()  # Set model to evaluate mode

            running_loss = 0.

            # Iterate over data.
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

                # statistics
                running_loss += loss_value.item()

            epoch_loss = running_loss / len(dataloader)

            logger.info(f'{phase} Loss: {epoch_loss}')
            if phase == 'train':
                wandb.log({"train loss": epoch_loss})
            else:
                threshold = 0.1
                mean_var_phi, accuracy_phi, mean_var_psi, accuracy_psi = angle_metrics(preds, targets, lengths,
                                                                                       threshold=threshold)
                wandb.log({"test loss": epoch_loss})
                wandb.log({"Mean phi absolute error": mean_var_phi})
                wandb.log({"Mean psi absolute error": mean_var_psi})
                wandb.log({f"Accuracy phi (threshold = {threshold})": accuracy_phi})
                wandb.log({f"Accuracy psi (threshold = {threshold})": accuracy_psi})
                threshold = 0.5
                mean_var_phi, accuracy_phi, mean_var_psi, accuracy_psi = angle_metrics(preds, targets, lengths,
                                                                                       threshold=threshold)
                wandb.log({f"Accuracy phi (threshold = {threshold})": accuracy_phi})
                wandb.log({f"Accuracy psi (threshold = {threshold})": accuracy_psi})

    return model


def get_logger():
    logger = logging.getLogger('Basic model train')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fileHandler = logging.FileHandler("rnn2.log")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    # 'application' code
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')

    return logger


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seq = get_embeddings(path_to_model, path_to_seq)
    angles = get_angles(path_to_angles)

    train_data, test_data = get_dataset(seq, angles)
    train_dataloader, val_dataloader = get_dataloaders(train_data, test_data, batch_size)

    logger = get_logger()
    wandb.init(project="antibodies-structure-prediction",
               name=f"basic-model n_layers={n_layers} batch_size={batch_size}")

    my_rnn = SimpleRNN(100, 4, 20, n_layers)
    my_rnn.to(device)
    wandb.watch(my_rnn)

    lr = 0.001
    num_epochs = 1000
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(my_rnn.parameters(), lr=lr)

    train_model(train_dataloader, val_dataloader, my_rnn, loss, optimizer, num_epochs, logger, device)

    torch.save(my_rnn.state_dict(), os.path.join(wandb.run.dir, 'my_rnn.pt'))


if __name__ == '__main__':
    main()
