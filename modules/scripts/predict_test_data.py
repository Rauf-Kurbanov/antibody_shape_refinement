from collections import OrderedDict

import torch
import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from models.simple_architecture.model import SimpleCharRNNUnit, SimpleCharRNN, PSEModel
from config_loader import load_config
from models.simple_architecture.simplemodel_coordinates.train import ComplexLoss
from metrics.metric import coordinate_metrics
from models.simple_architecture.data_utils import (collate_f,
                                                   get_full_data_coordinates,
                                                   get_sequence_data)

config = load_config('../')
MODELS_NAMES = ['simple', 'simple_coordinates']


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='simple_coordinates',
                        choices=MODELS_NAMES,
                        help='Which model to run')
    parser.add_argument('--saved_model_path', type=str, required=True,
                        help='Path to saved model')
    parser.add_argument('--input_size', type=int, required=True,
                        help='Model input size')
    parser.add_argument('--output_size', type=int, required=True,
                        help='Model output size')
    parser.add_argument('--hidden_dim', type=int, required=True,
                        help='Model hidden dim')
    parser.add_argument('--n_layers', type=int, required=True,
                        help='Model layers number')
    parser.add_argument('--data_path_seq', type=str, required=True,
                        help='Path to data file with embedded sequences')
    parser.add_argument('--data_path_coord', type=str, required=True,
                        help='Path to data file with target coordinates')
    parser.add_argument('--test_data_ids_path', type=str, required=True,
                        help='Path to test data file ids of pdb')
    parser.add_argument('--use_corrector', type=boolean_string, required=True,
                        help='Use corrector for coordinates?')
    return parser


BATCH_SIZE = 500


def load_test_data_to_dataloader(args):
    seq, coordinates = get_sequence_data(args['data_path_seq'], args['data_path_coord'])
    ids = []
    with open(args['test_data_ids_path'], 'r') as file:
        for line in file.readlines():
            ids.append(line[2:-2])
    seq = {bytes(id, 'utf-8'): seq[bytes(id, 'utf-8')] for id in ids}
    coordinates = {id: coordinates[id] for id in ids}
    test_data = [[x[1], x[2], x[3], x[0]] for x in get_full_data_coordinates(seq, coordinates)]
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_f)
    return test_dataloader


def prepare_padded_tensors(preds, targets, lengths):
    prepared = []
    for pred, target, length in zip(torch.unbind(preds), torch.unbind(targets), lengths):
        prepared.append((pred[:length, :], target[:length, :]))
    return prepared


def predict_test(model, test_data):
    loss = ComplexLoss(on_cpu=True)
    tensors_to_save = {}
    with torch.no_grad():
        for inputs, targets, lengths, names in test_data:
            preds, lengths, hiddens = model(inputs, lengths)
            loss_value = loss(preds, targets, lengths).item()
            print(f'Loss value: {loss_value}')
            metrics = coordinate_metrics(preds, targets, lengths, on_cpu=True)
            print(metrics)
            # bug if test data bigger batch size
            for name, item in zip(names, prepare_padded_tensors(preds, targets, lengths)):
                tensors_to_save[name] = item
    return tensors_to_save


def save_tensors(tensors_to_save, path=None):
    test_result_folder = '../'
    test_result_folder += config['PATH_TO_TEST_RESULTS'] if path is None else path
    Path(test_result_folder).mkdir(parents=True, exist_ok=True)
    for i, pt in tensors_to_save.items():
        i = i.decode('utf-8')
        torch.save(pt[0], test_result_folder + f'{i}_pred.pt')
        torch.save(pt[1], test_result_folder + f'{i}_target.pt')


if __name__ == '__main__':
    parser = get_parser()
    args = vars(parser.parse_known_args()[0])
    model_arg = args['model']
    model = None

    if model_arg == 'simple_coordinates':
        input_size = args['input_size']
        output_size = args['output_size']
        hidden_dim = args['hidden_dim']
        n_layers = args['n_layers']
        use_corrector = args['use_corrector']
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = SimpleCharRNN(1024, 9, 256, 1, device,
                              bilstm=True)

        saved_model_path = args['saved_model_path']
        state_dict = torch.load(saved_model_path, map_location=lambda storage, loc: storage)
        if isinstance(state_dict, OrderedDict):
            model.load_state_dict(state_dict)
        else:
            model = state_dict
        model.eval()
        model.use_corrector(use_corrector)
        model.cpu()
        model.device = torch.device('cpu')
        test_dataloader = load_test_data_to_dataloader(args)

        tensors_to_save = predict_test(model, test_dataloader)
        save_tensors(tensors_to_save)
    else:
        raise NotImplemented('Support only for simple model predicting coordinates.')
