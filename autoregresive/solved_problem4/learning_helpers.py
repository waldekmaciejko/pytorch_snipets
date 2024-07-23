import re

import matplotlib.pyplot as plt
import numpy as np
import os
import glob

import torch
import torch.nn as nn

import series_definition
import data_generators
import RNNmodel

def train_step_on_tensor(model: torch.nn.Module,
                           data: tuple,
                           loss_fn: torch.nn.Module,
                           optim: torch.optim.Optimizer,
                           device: str):
    """

    :param model:
    :param data: two dim tuple of tensors - batch first=False
    :param loss_fn:
    :param optim:
    :param device:
    :return:
    """

    y_predicted_train = []
    loss_train = 0

    for X_train_framed, y_train_framed in zip(data[0], data[1]):
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

    return loss_train, y_predicted_train

def train_step_on_dataset():

    pass

def test_step_on_tensor(model: torch.nn.Module,
                            data: tuple,
                            device: str,
                            loss_fn: torch.nn.Module):

    y_predicted_test = []
    loss_test = 0

    with torch.inference_mode():
        for X_test_framed, y_test_framed in zip(data[0], data[1]):
            model.eval()
            X_test_framed_device = X_test_framed.to(device)
            y_test_framed_device = y_test_framed.to(device)
            y_test_predict = model(X_test_framed_device)
            loss_te = loss_fn(y_test_predict[0], y_test_framed_device.unsqueeze(0))
            loss_test += loss_te.item()
            y_predicted_test.append(y_test_predict.item())

        return loss_test/len(data[1]), y_predicted_test, loss_te


def test_step_on_dataset():
    pass

def load_last_checkpoint(dir_to_checkpoints: str,
                         model: torch.nn.Module,
                         optim: torch.optim.Optimizer):

    m = {}
    if os.path.isfile(dir_to_checkpoints):
        print(f'Model is used: {dir_to_checkpoints}')

        checkpoint = torch.load(dir_to_checkpoints)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']

        m = {'model': model,
             'optim': optim,
             'loss': loss,
             'epoch': epoch
        }

    else:
        list1 = glob.glob((dir_to_checkpoints + "/*.pt"), recursive=True)
        list2 = [os.path.basename(i) for i in list1]
        pattern = re.compile('[0-9]+')

        try:
            list3 = np.array([int(re.search(pattern, i).group()) for i in list2])
        except TypeError:
            print("I can't find proper model index")

        list4 = np.argsort(list3)
        p = list1[list4[-1]]

        print(f'Checkpoint {p} was used')

        checkpoint = torch.load(p)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']

        m = {'model': model,
             'optim': optim,
             'loss': loss,
             'epoch': epoch
        }

    return m

def predict(model: torch.nn.Module,
            data: torch.Tensor,
            term_to_predict: int,
            device: str):



    #framed_data = data_generators.framing(data, term_to_predict, inc=1)

    y_predicted_test = []
    loss_test = 0

    with torch.inference_mode():
        for X_test_framed, y_test_framed in data[:2]:
            model.eval()
            X_test_framed_device = X_test_framed.to(device)
            y_test_framed_device = y_test_framed.to(device)
            y_test_predict = model(X_test_framed_device)
            loss_te = loss_fn(y_test_predict[0], y_test_framed_device.unsqueeze(0))
            loss_test += loss_te.item()
            y_predicted_test.append(y_test_predict.item())

        return loss_test/len(data[1]), y_predicted_test, loss_te

    model(data)


if __name__ == "__main__":

    # ======= signal to predict
    time_x = torch.linspace(0, 999, 800)
    i = 0.999
    tseries = series_definition.sinus_generator_v2(i, time_x)

    # predict next time_to_predict samples
    time_to_predict = 100

    # ======= call the appropriate model
    input_future = 1
    hidden_size = 50
    output_size = 1

    model = RNNmodel.LSTMmodel(input_future, hidden_size, output_size)
    loss_fn = nn.MSELoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=0.01)

    checkpoint = load_last_checkpoint(dir_to_checkpoints="../tSeriesResume/exp/model_8_cpu.pt",
                                                       model=model,
                                                       optim=optim)

    model = checkpoint['model']
    optim = checkpoint['optim']

    data = tseries[-time_to_predict:]

    # ======= create stepping loop
    with torch.inference_mode():
        for i in range(time_to_predict):
            model.eval()
            data = data[-time_to_predict:]
            y_predicted = model(data)
            data = torch.cat((data, y_predicted[0]), dim=0)

            stop = 0


    fix, ax = plt.subplots()
    ax.plot(range(len(tseries)), tseries)
    ax.plot(range(len(tseries) + time_to_predict)[-time_to_predict:], data[-time_to_predict:].detach().numpy())
    plt.show()


    stop = 0



