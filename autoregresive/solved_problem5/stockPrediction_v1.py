import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn

data = pd.read_csv("/home/ubuntu13042022/BADANIA/stockPrediction/Google_Stock_Price_Train.csv")
# data2 = pd.read_csv("/home/ubuntu13042022/BADANIA/stockPrediction/Google_Stock_Price_Test.csv")
# data = [data, data2]
print(data.head())
# priceOpen = data['Open'].values
print(data['Open'].info())
print(data['High'].info())
print(data['Low'].info())
print(data['Close'].info())
data['Close'].replace(',', '', inplace=True, regex=True)
#xx = np.array(data['Close']).astype(float)

#x = (data['Close'].apply(lambda x: True if ',' in x else False))
#fig, ax = plt.subplots()
#ax.plot(data['Close'])
#ax.plot(xx.T)
#plt.show()

price = data[['Close']].values

print(type(price))
# print(price.head())
# print(price.info())

scaler = MinMaxScaler(feature_range=(-1, 1))
price = scaler.fit_transform(price.reshape(-1, 1))

def split_data(stock, lookback):
    data_raw = stock
    data_in_loop = []

    for index in range(len(data_raw) - lookback):
        data_in_loop.append(data_raw[index: index + lookback])

    data_in_loop = np.array(data_in_loop)
    test_set_size = int(np.round(0.2 * data_in_loop.shape[0]))
    train_set_size = data_in_loop.shape[0] - (test_set_size)

    #fig, ax = plt.subplots()
    #for i in range(data.shape[1]):
        #ii = data[0, i, :]
        #ax.plot(data[:, i, :])

    x_train = data_in_loop[:train_set_size, :-1, :]
    y_train = data_in_loop[:train_set_size, -1, :]

    x_test = data_in_loop[train_set_size:, :-1]
    y_test = data_in_loop[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test, data_in_loop]

lookback = 20

x_train, y_train, x_test, y_test, data_in_loop = split_data(price, lookback)

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)

y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)

y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out

model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

import time

hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []

for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_lstm)
    print(f"Epochs: {t} MSE: {loss.item()}")
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

training_time = time.time() - start_time
print("Training time: {}".format(training_time))

predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))

fig, ax = plt.subplots(2, 1)
ax[0].plot(predict)
ax[0].plot(original)

ax[1].plot(hist)

plt.show()

stop = 0
