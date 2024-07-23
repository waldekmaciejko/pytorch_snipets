## IT'S WORK !!!! - focous on that
## to correct train loop
"""
Example of use autoregresive recurent neural network model to predict
feature values of sequance

The goal is to predict window length feature data of sequance
"""
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import data_generators
import series_definition
import RNNmodel
import early_stopping
import learning_helpers

from sklearn.metrics import mean_squared_error

# ========== prepare synthetic data
test_size = 200
seq_len = 800
window_len = 100
inc = 1

sinus_generator = series_definition.sinus_generator_v2
data = data_generators.synthetic_data_generator_v4(sinus_generator, seq_len=seq_len, window_len=window_len, inc=inc)
X_data = data[0]
y_data = data[1]

# ========== split data
X_train = X_data[:-test_size]
y_train = y_data[:-test_size]

fig, ax = plt.subplots()
ax.plot(range(len(X_train[-2])), X_train[-2])
plt.show()

X_test = X_data[-test_size:]
y_test = y_data[-test_size:]

# ========== define model
input_future = 1
hidden_size = 50
output_size = 1

model = RNNmodel.LSTMmodel(input_future, hidden_size, output_size)
loss_fn = nn.MSELoss()
optim = torch.optim.SGD(params=model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.9, end_factor=0.01, total_iters=10)
early_stopping = early_stopping.EarlyStopping(wait=8, delta=0e-5, path="../tSeriesResume/exp")

checkpoint = None
use_checkpoint = False

if use_checkpoint:
    checkpoint = learning_helpers.load_last_checkpoint(dir_to_checkpoints="../tSeriesResume/exp",
                                                       model=model,
                                                       optim=optim)

    model = checkpoint['model']
    optim = checkpoint['optim']

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

loss_te = 0

for i in range(n_epochs):

    loss_train, y_predicted_train = learning_helpers.train_step_on_tensor(model=model,
                                                                          data=(X_train, y_train),
                                                                          loss_fn=loss_fn,
                                                                          optim=optim,
                                                                          device=device)
    scheduler.step()

    loss_test, y_predicted_test, loss_te = learning_helpers.test_step_on_tensor(model=model,
                                                                                data=(X_test, y_test),
                                                                                loss_fn=loss_fn,
                                                                                device=device)

    if i % 1 == 0:
        ax.plot(data[2][-test_size:], np.array(y_predicted_test).squeeze(), label=f'epoch: {i}')
        print(f"Epoch {i}| on train: loss={loss_train/len(y_train):.2f} "
              f"MSE={mean_squared_error(y_train, y_predicted_train):.2f}| "
              f"on test: loss={loss_test/len(y_test):.2f}, MSE={mean_squared_error(y_test, y_predicted_test):.2f}| "
              f"lr={optim.param_groups[0]['lr']:.5f}")

    early_stopping(name_of_model=f'model_{i}_{device}.pt',
                   val_loss=loss_te,
                   model=model,
                   optim=optim,
                   epoch=i)

    if early_stopping.early_stop:
        print('Early stopping')
        break

if ~early_stopping.early_stop:
    early_stopping.save_check_point(val_loss=loss_te,
                                    model=model,
                                    optim=optim,
                                    epoch=n_epochs)

    print('Checkpoint saved after all epochs')

ax.legend()
plt.show()

