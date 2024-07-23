import torch
import matplotlib.pyplot as plt
import numpy as np
import random

def sinus_generator(i, time_x):

    #signal = torch.as_tensor(i) + torch.sin(4 * time_x * torch.pi * i)
    signal = 1000 * np.sin(time_x)
    noise = np.random.normal(scale=30, size=time_x.shape[0])
    distorted_signal = torch.as_tensor(signal + noise, dtype=torch.float)
    return distorted_signal

def sinus_generator_v2(i, time_x):

    signal = torch.as_tensor(i) + torch.sin(4 * time_x * torch.pi * i)
    return signal

def sinus_generator_v3(i, time_x):

    signal = torch.sin(time_x * 2 * torch.pi/40 * i)
    return signal


if __name__ == "__main__":

    torch.random.manual_seed(42)

    time_x = torch.linspace(0, 80, 800)
    i = random.uniform(0.998, 0.999)

    y_val = sinus_generator(i, time_x)

    fig, ax = plt.subplots()
    ax.plot(time_x, y_val)
    plt.show()

