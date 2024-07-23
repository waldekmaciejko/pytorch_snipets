# pytorch_snipets

Practical use of basic tool for machine learning using PyTorch lib.

---

recapAllInOne.py

---

- example of binary classification cats vs dogs
- to start tensorboard ...$ tensorboard --logdir=logs
- reads data from one, local source (no split into traning and validaion dataset) using lista and torch.utils.data.Dataset
- uses torchvision to transform data
- uses custom function in torchvision.transform pipline
- estimate accuracy (sigmoid) (no nn.BCEWithLogitsLoss)
- added the adaptive LR
- added early stopping and dropout layer
- added argument parser

---

autoregresive/\*

---

- contains several easy solutions of common problems of predictions time series using autoregresive method
- framing function
- Vanilla RNN
