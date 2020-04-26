import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.rnn as rnn_utils


class SimpleRNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(SimpleRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True,
                            bidirectional=True)

        self.linear1 = nn.Linear(2 * hidden_dim, 2 * output_size)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(2 * output_size, output_size)

    def forward(self, x, lengths, hiddens=None):
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True)
        x, hiddens = self.lstm(x, hiddens)
        x, _ = rnn_utils.pad_packed_sequence(x, batch_first=True)

        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x, lengths, hiddens


def rescale_prediction(pred):
    d1 = 1.4555
    d2 = 1.5223
    d3 = 1.3282
    ds = [d1, d3, d2]
    d = torch.tensor(ds * len(pred)).to(pred.device)
    d = d[:len(pred) - 1]
    x1 = pred[:-1]
    x2 = pred[1:]
    alpha = d / (x1 - x2).norm(dim=-1)
    alpha = torch.unsqueeze(alpha, 1).repeat(1, 3)
    old_diff = x2 - x1
    old_diff[old_diff.norm(dim=-1) < 0.001] = 0
    diff = old_diff * alpha
    pred_new = pred[0].unsqueeze(0)
    for i in range(len(diff)):
        pred_new = torch.cat((pred_new, pred_new[-1] + diff[i].unsqueeze(0)), 0)
    return pred_new


class SimpleCharRNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers, device, bilstm=False):
        super(SimpleCharRNN, self).__init__()
        self.bilstm = bilstm
        self.device = device
        if self.bilstm:
            self.model_forward = SimpleCharRNNUnit(input_size, output_size, hidden_dim, n_layers, device)
            self.model_backward = SimpleCharRNNUnit(input_size, output_size, hidden_dim, n_layers, device)
        else:
            self.model = SimpleCharRNNUnit(input_size, output_size, hidden_dim, n_layers, device)

    def reverse_tensor(self, x):
        x = x.permute(1, 0, 2)
        idx = [i for i in range(x.size(0) - 1, -1, -1)]
        idx = torch.LongTensor(idx).to(self.device)
        inverted_tensor = x.index_select(0, idx)
        inverted_tensor = inverted_tensor.permute(1, 0, 2)
        return inverted_tensor

    def use_corrector(self, use):
        if self.bilstm:
            self.model_forward.use_corrector(use)
            # self.model_backward.use_corrector(use)
        else:
            self.model.use_corrector(use)

    def forward(self, x, lengths, hiddens=None):
        if self.bilstm:
            answer_b, lengths_b, z_b = self.model_backward(self.reverse_tensor(x), lengths, hiddens)
            answer_f, lengths_f, z_f = self.model_forward(x, lengths, hiddens, self.reverse_tensor(answer_b))
            # answer, lengths, z = answer_f + answer_b / 2, lengths_f, z_f
            answer, lengths, z = answer_f, lengths_f, z_f
            return answer, lengths, z
        else:
            return self.model(x, lengths, hiddens)


class SimpleCharRNNUnit(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers, device):
        super(SimpleCharRNNUnit, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.device = device

        self.lstmcell = nn.LSTMCell(input_size + 9, hidden_dim)
        # self.h2h = nn.Linear(hidden_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_size)
        # self.h2o = nn.Sequential(nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(inplace=True),
        #                          nn.Linear(hidden_dim, output_size))
        self.corrector_i = 0
        self.use_corrector_flag = False

    def init_hiddens(self, batch):
        h0 = torch.zeros(batch, self.hidden_dim).requires_grad_()
        nn.init.kaiming_uniform_(h0)

        # Initialize cell state
        c0 = torch.zeros(batch, self.hidden_dim).requires_grad_()
        nn.init.kaiming_uniform_(c0)
        return (h0, c0)

    def use_corrector(self, use):
        self.use_corrector_flag = use

    def init_preds(self, batch):
        return torch.zeros((batch, self.output_size))

    def correct_coordinates(self, coordinates, finished):
        coordinates_l = list(coordinates.unbind(0))
        for i, c in enumerate(coordinates_l):
            c = c.view(-1, 3)
            if i not in finished:
                coordinates_l[i] = rescale_prediction(c)
            else:
                coordinates_l[i] = c
        coordinates = torch.stack(coordinates_l)

        return coordinates

    def forward(self, x, lengths, hiddens=None, y=None):
        batch_size = x.shape[0]
        h, c = self.init_hiddens(batch_size)
        answer = torch.FloatTensor([])
        output_i = torch.zeros((x.size(0), 9))
        output_i_prev = None
        h = h.to(self.device)
        c = c.to(self.device)
        answer = answer.to(self.device)
        output_i = output_i.to(self.device)

        for i, input_i in enumerate(x.chunk(x.size(1), dim=1)):
            input_i = torch.cat((input_i, output_i.unsqueeze(1)), dim=-1)
            h, c = self.lstmcell(input_i.squeeze(1), (h, c))
            output_i = self.h2o(h)
            if y is not None:
                output_i = (output_i + y[:, i, :]) / 2
            if self.use_corrector_flag and not self.training:
                if output_i_prev is not None:
                    last_two = torch.cat((output_i_prev.unsqueeze(1), output_i.unsqueeze(1)), dim=1)
                    _, output_i = self.correct_coordinates(last_two,
                                                           np.arange(len(lengths.cpu()))[lengths.cpu() <= i]) \
                        .chunk(2, 1)
                    output_i[i >= lengths] = torch.zeros_like(output_i[0])
                    output_i = output_i.view(-1, 9)
                else:
                    output_i = self.correct_coordinates(output_i.unsqueeze(1), np.array([]))
                    output_i = output_i.view(-1, 9)
            output_i_prev = output_i.view(-1, 9).clone()
            answer = torch.cat((answer, output_i.unsqueeze(1)), dim=1)

        return answer, lengths, None
