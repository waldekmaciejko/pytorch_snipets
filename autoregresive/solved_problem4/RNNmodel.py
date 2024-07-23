import torch
import torch.nn as nn

class LSTMmodel(nn.Module):
    def __init__(self, input_future, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.rnn_block1 = nn.Sequential(
            nn.LSTM(input_future, hidden_size)
        )

        self.linear_block2 = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )
        self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))

    def forward(self, sequence):
        output, self.hidden = self.rnn_block1(sequence.reshape(-1, 1, 1))
        x = self.linear_block2(output)
        return x[-1]




