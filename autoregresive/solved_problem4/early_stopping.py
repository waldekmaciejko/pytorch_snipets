import torch
import numpy as np
from datetime import datetime
import os


# thanks to Bjarten/early-stopping-pytorch

class EarlyStopping:
    """

    """

    def __init__(self, wait=8, delta=0e-5, path="../tSeriesResume/exp"):

        self.wait = wait
        self.counter = 0
        self.best_score = 0
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.val_loss_min = np.Inf

    def __call__(self, name_of_model: str, val_loss, model, optim, epoch):
        score = val_loss
        self.path_to_save = os.path.join(self.path, name_of_model)

        if self.best_score == 0:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.wait:
                self.early_stop = True
                self.save_check_point(val_loss, model, optim, epoch)
        else:
            self.best_score = score
            self.counter = 0

    def save_check_point(self,
                         val_loss: torch.Tensor,
                         model: torch.nn.Module,
                         optim: torch.optim,
                         epoch: int):

        print(f"Check point on {datetime.now()}, saved in {self.path_to_save}")
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': val_loss},
                   self.path_to_save)
        self.val_loss_min = val_loss
