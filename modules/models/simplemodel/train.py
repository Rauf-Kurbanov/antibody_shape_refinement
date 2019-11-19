import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader

from modules.embedding import get_embeddings
from modules.embedding import get_angles

path_to_model = 'data/embedding/pretrained_models/ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav'
path_to_data = 'data/cdr_h3_seq'


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


def get_dataloaders(train_dataset, test_dataset):
    batch_size = 200
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

    val_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

    return train_dataloader, val_dataloader


def main():
    seq = get_embeddings(path_to_model, path_to_data)
    angles = get_angles(path_to_data)

    train_data, test_data = get_dataset(seq, angles)
    train_dataloader, val_dataloader = get_dataloaders(train_data, test_data)


if __name__ == '__main__':
    main()
