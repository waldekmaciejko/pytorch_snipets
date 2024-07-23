import numpy as np
import torch
import random


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
    enframed = framing(y_values, window, inc)
    label = enframed[:, -1][1:]
    # last window has to be omit (no next label)
    data = enframed[:-1]
    return (data, label, time_x, y_values)

def synthetic_data_generator_v4(f_series_shape, seq_len=6, window_len=2, inc=1):

    window = window_len
    # rate = [random.uniform(0.998, 0.999) for i in range(n_random)]
    ### time_x = torch.linspace(0, 999, seq_len)
    data_list = []
    label_list = []
    # for i in rate:
    i = random.uniform(0.998, 0.999)

    time_x = torch.linspace(0, 799, seq_len)
    #y_values = torch.sin(time_x * 2 * torch.pi / 40)
    ##y_values = torch.as_tensor(i) + torch.sin(4 * time_x * torch.pi * i)
    y_values = f_series_shape(i, time_x)


    # take the last column of y_values as a label
    # (value after each frame is a label),
    # y_values at time=0 has to omit (no previous frame)
    enframed = framing(y_values, window, inc)
    label = enframed[:, -1][1:]
    # last window has to be omit (no next label)
    data = enframed[:-1]

    return (data, label, time_x, y_values)