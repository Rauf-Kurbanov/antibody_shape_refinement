import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.rnn as rnn_utils


class PSEModel(nn.Module):

    def __init__(self, model, path_to_emb_model):
        super(PSEModel, self).__init__()
        encoder = torch.load(path_to_emb_model)
        encoder = encoder.embedding
        self.emb = encoder
        self.model = model

    def embed_data(self, x):
        if self.emb.lm:
            h = self.emb.embed(x)
        else:
            if type(x) is rnn_utils.PackedSequence:
                h = self.emb.embed(x.data)
                h = rnn_utils.PackedSequence(h, x.batch_sizes)
            else:
                h = self.emb.embed(x)

        h, _ = self.emb.rnn(h)

        if type(h) is rnn_utils.PackedSequence:
            h = h.data
            h = self.emb.dropout(h)
            z = self.emb.proj(h)
            z = rnn_utils.PackedSequence(z, x.batch_sizes)
        else:
            h = h.reshape(-1, h.size(2))
            h = self.emb.dropout(h)
            z = self.emb.proj(h)
            z = z.reshape(x.size(0), x.size(1), -1)

        return z

    def forward(self, x, lengths, hiddens=None):
        x = self.embed_data(x)
        x = self.model(x, lengths, hiddens)

        return x


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
            x_len = len(x.chunk(x.size(1), dim=1))
            if i in [0, 1, 2] or i in [x_len - 4, x_len - 3, x_len - 2, x_len - 1]:
                pass
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


class SimpleRNNSphere(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers, device, bilstm=True):
        super(SimpleRNNSphere, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.device = device
        self.bilstm = bilstm

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True, bidirectional=self.bilstm)
        self.h2o = nn.Linear(2 * hidden_dim if self.bilstm else hidden_dim, output_size)
        self.tanh = nn.Tanh()

    def get_vector(self, d, angles):
        theta = angles[:, 0]
        phi = angles[:, 1]
        x = d * torch.sin(theta) * torch.cos(phi)
        y = d * torch.sin(theta) * torch.sin(phi)
        z = d * torch.cos(theta)
        return (x, y, z)

    def get_coordinates(self, angles, lengths):
        d1 = 1.4555
        d2 = 1.5223
        d3 = 1.3282
        d = [d1, d2, d3]
        point = torch.zeros((angles.shape[0], 3)).to(angles.device)
        result = []
        first = True
        for i in range(angles.shape[1]):
            for j in range(3):
                if first:
                    first = False
                    result.append(point)
                else:
                    k = 2 * j
                    vector = self.get_vector(d[(j + 2) % 3], angles[:, i, k:k + 2])
                    point = point + torch.cat(vector).view(point.shape)
                    point[lengths >= i] = torch.FloatTensor((0, 0, 0)).to(point.device)
                    result.append(point)
        result = torch.stack(result).view(angles.shape[0], -1, 9)
        return result

    def forward(self, x, lengths, hiddens=None, y=None):
        (h, c) = self.lstm(x)
        x = self.h2o(h)
        x = self.tanh(x)
        angles = torch.atan2(x[:, :, :6], x[:, :, 6:])
        angles[:, :, 3:] += np.pi
        angles[:, :, 3:] /= 2
        answer = self.get_coordinates(angles, lengths)
        return answer, lengths, None


class SimpleCNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, kernel_size, device, dropout=0.2):
        super(SimpleCNN, self).__init__()

        self.hidden_dim = 128
        self.output_size = output_size
        self.n_layers = 0
        self.kernel_size = kernel_size
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        self.device = device

        self.in2hid = nn.Linear(input_size, hidden_dim)

        self.convs_hidden = nn.ModuleList([nn.Conv1d(in_channels=hidden_dim,
                                                     out_channels=2 * hidden_dim,
                                                     kernel_size=kernel_size,
                                                     padding=(kernel_size - 1) // 2)
                                           for _ in range(self.n_layers)])
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=128,
                                              out_channels=64,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2),
                                    nn.Conv1d(in_channels=64,
                                              out_channels=32,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2),
                                    nn.Conv1d(in_channels=32,
                                              out_channels=16,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2),
                                    nn.Conv1d(in_channels=16,
                                              out_channels=9,
                                              kernel_size=kernel_size,
                                              padding=(kernel_size - 1) // 2)
                                    ])

        self.dropout = nn.Dropout(dropout)

        self.lstm_out = SimpleCharRNN(9, 9, 16,
                                      1, device, bilstm=True)
        self.fc_out = nn.Linear(9, output_size)

    def forward(self, x, lengths, hiddens=None):
        conv_input = self.in2hid(x)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs_hidden):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            # conved = [batch size, 2 * hid dim, src len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim, src len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # conved = [batch size, hid dim, src len]

            # set conv_input to conved for next loop iteration
            conv_input = conved
        for i, conv in enumerate(self.convs):
            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            # conved = [batch size, 2 * hid dim, src len]

            # pass through GLU activation function
            # conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim, src len]

            # apply residual connection
            # conved = (conved + conv_input) * self.scale

            # conved = [batch size, hid dim, src len]

            # set conv_input to conved for next loop iteration
            conv_input = conved
        conved = conved.permute(0, 2, 1)
        answer, _, _ = self.lstm_out(conved, lengths)
        answer = self.fc_out(answer)
        return answer, lengths, None
