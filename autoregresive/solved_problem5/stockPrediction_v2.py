import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch


#torch.hub('NVIDIA/DeepLearningExamples:torchhub', )


data = pd.read_csv("/home/ubuntu13042022/BADANIA/stockPrediction/Google_Stock_Price_Train.csv")

data['Close'].replace(',', '', regex=True, inplace=True)
x = data['Close'].values.astype(float)

sc = MinMaxScaler(feature_range=(-1, 1))
x_scaled = sc.fit_transform(x.reshape(-1, 1))


def sliding_window(x_scaled_: np.ndarray,
                   n_window_size: int,
                   train_size: float) -> (np.ndarray, np.ndarray):
    x_ = []
    y = []
    for i in range(len(x_scaled_) - n_window_size):
        x_.append(x_scaled_[i:(i + n_window_size), :])
        y.append(x_scaled_[n_window_size + i, :])

    #        stop = 0

    x_train_, x_test_, y_train_, y_test_ = train_test_split(x_,
                                                            y,
                                                            train_size=train_size,
                                                            random_state=42,
                                                            shuffle=True)

    return np.array(x_train_), np.array(x_test_), np.array(y_train_), np.array(y_test_)


X_train, X_test, y_train, y_test = sliding_window(x_scaled, 10, 0.8)

X_train_torch = torch.from_numpy(X_train).type(torch.Tensor)
X_test_torch = torch.from_numpy(X_test).type(torch.Tensor)

y_train_torch = torch.from_numpy(y_train).type(torch.Tensor)
y_test_torch = torch.from_numpy(y_test).type(torch.Tensor)


class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_, n_ouputs):
        super(VanillaRNN, self).__init__()
        self.n_features = input_dim
        self.hidden_dim = hidden_dim_
        self.n_ouputs = n_ouputs
        self.hidden = None

        self.basic_rnn = nn.RNN(self.n_features, self.hidden_dim, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, self.n_ouputs)

    def forward(self, x_):
        batch_first_output, self.hidden = self.basic_rnn(x_)
        last_output = batch_first_output[:, -1]
        out = self.classifier(last_output)
        return out.view(-1, self.n_ouputs)


input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

model = VanillaRNN(input_dim=input_dim,
                 hidden_dim_=hidden_dim,
                 n_ouputs=output_dim)

criterion = torch.nn.MSELoss(reduction='mean')
optimize = torch.optim.Adam(model.parameters(), lr=0.01)

hist = np.zeros(num_epochs)
rnn_ = []

for i in range(num_epochs):
    y_train_pred = model(X_train_torch)
    loss_value = criterion(y_train_pred, y_train_torch)
    hist[i] = loss_value.item()

    optimize.zero_grad()
    loss_value.backward()
    optimize.step()

predict = pd.DataFrame(sc.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(sc.inverse_transform(y_train_torch.detach().numpy()))

fig, ax = plt.subplots(2, 1)
ax[0].plot(predict)
ax[0].plot(original)

ax[1].plot(hist)

plt.show()

stop = 0
