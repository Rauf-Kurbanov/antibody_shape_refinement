import pickle
import numpy as np
import torch

from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from sklearn.model_selection import train_test_split


def pad_data_collate(X, y, lengths, max_length=None):
    if max_length is not None:
        X = [x[:max_length] for x in X]
        y = [y[:max_length] for y in y]
        lengths = [np.minimum(l, max_length) for l in lengths]
    y_padded = rnn_utils.pad_sequence(y, batch_first=True)
    X_padded = rnn_utils.pad_sequence(X, batch_first=True)
    X, y, lengths = zip(*[[X_padded[i], y_padded[i], lengths[i]] for i in range(len(X_padded))])
    X = torch.stack(X)
    y = torch.stack(y)
    lengths = torch.tensor(lengths)
    return X, y, lengths


def pad_data(X_train, y_train, X_test, y_test, lengths_train, lengths_test, max_length=None):
    if max_length is not None:
        X_train = [x[:max_length] for x in X_train]
        X_test = [x[:max_length] for x in X_test]
        y_train = [y[:max_length] for y in y_train]
        y_test = [y[:max_length] for y in y_test]
        lengths_train = [np.minimum(l, max_length) for l in lengths_train]
        lengths_test = [np.minimum(l, max_length) for l in lengths_test]
    y_train_padded = rnn_utils.pad_sequence(y_train, batch_first=True)
    X_train_padded = rnn_utils.pad_sequence(X_train, batch_first=True)
    train_dataset = [[X_train_padded[i], y_train_padded[i], lengths_train[i]] for i in range(len(X_train_padded))]

    X_test_padded = rnn_utils.pad_sequence(X_test, batch_first=True)
    y_test_padded = rnn_utils.pad_sequence(y_test, batch_first=True)
    test_dataset = [[X_test_padded[i], y_test_padded[i], lengths_test[i]] for i in range(len(X_test_padded))]

    return train_dataset, test_dataset


def get_dataset_angles(seq, angles):
    full_data = [[len(seq[x]), seq[x], angles[x]] for x in seq.keys()]
    test_data = full_data[:int(len(full_data) / 10)]
    train_data = full_data[int(len(full_data) / 10):]

    full_data.sort(key=lambda x: x[0], reverse=True)

    train_data.sort(key=lambda x: x[0], reverse=True)
    lengths_train = [train_data[i][0] for i in range(len(train_data))]
    X_train = [train_data[i][1] for i in range(len(train_data))]
    y_train = [train_data[i][2] for i in range(len(train_data))]

    test_data.sort(key=lambda x: x[0], reverse=True)
    lengths_test = [test_data[i][0] for i in range(len(test_data))]
    X_test = [test_data[i][1] for i in range(len(test_data))]
    y_test = [test_data[i][2] for i in range(len(test_data))]

    train_dataset, test_dataset = pad_data(X_train, y_train, X_test, y_test, lengths_train, lengths_test)

    return train_dataset, test_dataset


def get_dataset_coordinates(seq, coordinates):
    full_data = [[len(seq[x]), seq[x], coordinates[x.decode('ascii')]] for x in seq.keys()]
    train_data, test_data = train_test_split(full_data, test_size=0.1, shuffle=True)
    return train_data, test_data


def collate_f(batch):
    batch.sort(key=lambda x: x[0], reverse=True)
    lengths = [batch[i][0] for i in range(len(batch))]
    X = [batch[i][1] for i in range(len(batch))]
    y = [batch[i][2] for i in range(len(batch))]
    data = pad_data_collate(X, y, lengths, max_length=50)
    return data


def get_dataloaders(train_dataset, test_dataset, batch_size):
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_f)

    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_f)

    return train_dataloader, val_dataloader


def get_embedded_data_angles(config):
    with open(config["PATH_TO_SEQ_EMBEDDED"], 'rb') as handle:
        seq = pickle.load(handle)
    with open(config["PATH_TO_ANGLES_EMBEDDED"], 'rb') as handle:
        angles = pickle.load(handle)
    return seq, angles


def get_embedded_data_coordinates(config):
    with open(config["PATH_TO_SEQ_EMBEDDED"], 'rb') as handle:
        seq = pickle.load(handle)
    with open(config["PATH_TO_COORD_EMBEDDED"], 'rb') as handle:
        coordinates = pickle.load(handle)
    return seq, coordinates
