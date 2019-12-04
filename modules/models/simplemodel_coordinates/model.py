import torch.nn as nn
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
