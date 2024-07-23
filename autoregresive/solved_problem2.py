## IT'S WORK !!!!
"""
Example of use autoregresive recurent neural network model to predict
feature values of sequance

The aim is to predict window length feature data of sequance
"""
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error

class Tools:

    @staticmethod
    def framing(x: np.array, win: int, inc: int) -> np.array:
        """
        :param x: np.array [time_index, seq_length] - signal
        :param win: int - window length
        :param increment: int - window step
        :return:
        """
        nx = x.shape[0]

        nf = int(np.fix((nx - win + inc) / inc))
        f = np.zeros([nf, int(win)])
        indf = inc * (np.linspace(0, (nf - 1), nf))
        inds = np.linspace(0, (win - 1), win)

        a = indf.reshape(-1, 1).repeat(win, axis=1)
        b = inds.reshape(-1, 1).repeat(nf, axis=1)
        idx = np.array((a + b.T), dtype=int)
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
            return torch.as_tensor(x[idx].reshape(nf, win))
        else:
            return x[idx]

class SyntheticDataGenerator_v3:

    @staticmethod
    def synthetic_data_generator_v3(seq_len=6, window_len=2, inc=1):
        window = window_len
        # rate = [random.uniform(0.998, 0.999) for i in range(n_random)]
        ### time_x = torch.linspace(0, 999, seq_len)
        data_list = []
        label_list = []
        # for i in rate:
        i = random.uniform(0.998, 0.999)

        time_x = torch.linspace(0, 799, 800)
        #y_values = torch.sin(time_x * 2 * torch.pi / 40)
        y_values = torch.as_tensor(i) + torch.sin(4 * time_x * torch.pi * i)

        # take the last column of y_values as a label
        # (value after each frame is a label),
        # y_values at time=0 has to omit (no previous frame)
        enframed = Tools.framing(y_values, window, inc)
        label = enframed[:, -1][1:]
        # last window has to be omit (no next label)
        data = enframed[:-1]
        return (data, label, time_x, y_values)

class VanillaRNN(nn.Module):
    def __init__(self, input_future, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_future, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))

    def forward(self, sequence):
        output, h = self.rnn(sequence.reshape(-1, 1, 1))
        x = self.linear(output)
        return x[-1]


if __name__ == "__main__":
    
    train_dataset_size = 50
    test_dataset_size = 10
    seq_len = 60000
    window_len = 40
    inc = 1

    n_frames = int(np.floor((seq_len - window_len) / inc))
    #row_numbers = list(range(n_frames))

    # ========== Prepare synthetic data
    data = SyntheticDataGenerator_v3.synthetic_data_generator_v3(seq_len=seq_len, window_len=window_len, inc=inc)
    X_data = data[0]
    y_data = data[1]

    test_size = 40

    X_train = X_data[:-test_size]
    y_train = y_data[:-test_size]

    X_test = X_data[-test_size:]
    y_test = y_data[-test_size:]

    input_future = 1
    hidden_size = 50
    output_size = 1

    model = VanillaRNN(input_future, hidden_size, output_size)
    callbacks = []
    #model.set_callbacks(callbacks)
    loss_fn = nn.MSELoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=0.01)
    #scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.9, end_factor=0.01, total_iters=10)

    loss_train = 0
    loss_test = 0
    n_epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss = 0

    fig, ax = plt.subplots()
    ax.plot(data[2][:-test_size], data[3][:-test_size], 'g')
    ax.plot(data[2], data[3], 'o')

    y_predicted_train = []
    y_predicted_test = []

    for i in range(n_epochs):

        for X_train_framed, y_train_framed in zip(X_train, y_train):
            model.train()
            X_train_framed_device = X_train_framed.to(device)
            y_train_framed_device = y_train_framed.to(device)
            y_train_predict = model(X_train_framed_device)
            y_predicted_train.append(y_train_predict.item())
            loss_tr = loss_fn(y_train_predict[0], y_train_framed_device.unsqueeze(0))
            loss_tr.backward()
            optim.step()
            optim.zero_grad()
            loss_train += loss_tr.item()
        #scheduler.step()

        with torch.inference_mode():
            for X_test_framed, y_test_framed in zip(X_test, y_test):
                model.eval()
                X_test_framed_device = X_test_framed.to(device)
                y_test_framed_device = y_test_framed.to(device)
                y_test_predict = model(X_test_framed_device)
                loss_te = loss_fn(y_test_predict[0], y_test_framed_device.unsqueeze(0))
                loss_test += loss_te.item()
                y_predicted_test.append(y_test_predict.item())

            if i % 5 == 0:
                ax.plot(data[2][-test_size:], np.array(y_predicted_test).squeeze(), label=f'epoch: {i}')
                print(f"Epoch {i}| on train: loss={loss_train/len(y_train):.2f} "
                      f"MSE={mean_squared_error(y_train, y_predicted_train):.2f}| "
                      f"on test: loss={loss_test/len(y_test):.2f}, MSE={mean_squared_error(y_test, y_predicted_test):.2f}|")
            y_predicted_train = []
            y_predicted_test = []
            loss_train = 0
            loss_test = 0

    ax.legend()
    plt.show()

