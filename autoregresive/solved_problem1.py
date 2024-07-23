"""
Example of use autoregresive recurent neural network model to predict
feature values of sequance

The goal is to predict window length feature data of sequance
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Tools:

    @staticmethod
    def windowing(seq, ws):

        # windowing on signal function default with step=1
        # return list

        out = []
        L = len(seq)

        for _idx in range(L - ws):
            window = seq[_idx:_idx + ws]
            label = seq[_idx + ws:_idx + ws + 1]
            out.append((window, label))

        return out

    @staticmethod
    def generate_synthetic():
        # generate synthetic data
        x = torch.linspace(0, 799, 800)
        y = torch.sin(x * 2 * torch.pi / 40)

        return x, y

class vanillaLSTM(nn.Module):

    def __init__(self, input_size=1, hidden_size=50, out_size=1):

        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)
        self.hidden = (torch.zeros(1, 1, self.hidden_size),
                       torch.zeros(1, 1, self.hidden_size))

    def forward(self, seq):

        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1))
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]

if __name__ == "__main__":

    x, y = Tools.generate_synthetic()
    test_size = 40
    train_set = y[:-test_size]
    test_set = y[-test_size:]

    window_size = 40
    train_data = Tools.windowing(train_set, window_size)
    torch.manual_seed(42)
    model = vanillaLSTM()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 10
    future = 40

    fig, ax = plt.subplots()
    ax.set_xlim(700, 801)

    loss_train = []
    loss_test = []

    for i in range(epochs):
        for seq_test, y_train in tqdm(train_data, desc=f'Train, epoch {i}'):
            model.train()
            optimizer.zero_grad()
            y_pred = model(seq_test)
            loss = loss_fn(y_pred, y_train)
            loss_train.append(loss.item())
            loss.backward()
            optimizer.step()

        preds = train_set[-window_size:].tolist()

        for f in tqdm(range(future), desc=f'Test, epoch: {i}'):
            seq_test = torch.FloatTensor(preds[-window_size:])
            with torch.inference_mode():
                preds.append(model(seq_test).item())

            loss = loss_fn(torch.tensor(preds[-window_size:]), y[-window_size:])
            loss_test.append(loss.item())

        if i%5 == 0:
            print(f"Epoch: {i}, train loss: {np.mean(loss_train):.2f}, test loss: {np.mean(loss_test):.2f}")

        ax.plot(y.numpy())
        ax.plot(range(760, 800), preds[window_size:])

    plt.grid(True)
    plt.show()