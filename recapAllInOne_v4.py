# example of binary classification cats vs dogs
## to start tensorboard ...$ tensorboard --logdir=logs
# read data from one, local source using lista and torch.utils.data.Dataset
# use torchvision to transform data
# use custom function in torchvision.transform pipline
# estimate accuracy (sigmoid)
# added the adaptive LR
# added early stopping and dropout layer
# added argument parser
import matplotlib.pyplot as plt
# train
# python recapAllInOne_v4.py --path_to_dataset='data/cats_or_dogs'
#                           --n_epochs=10 --learning_rate=0.01 --batch_size=32
# predict
# python recapAllInOne_v4.py --predict
#                           --path_to_predict_model='exp/model_10_32_0.01_cuda_20240716_130308_4.pt'
#                           --path_to_image_to_predict='data/cats_or_dogs/Cat/0.jpg'

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from datetime import datetime
import glob
import os
import random
import numpy as np
from PIL import Image
import argparse


class CustomDataset(nn.Module):
    # not used - only

    def __init__(self, path, transform=None):

        self.paths_dataset = path
        self.labels = torch.as_tensor([0 if 'Cat' in l.split('/') else 1 for l in self.paths_dataset], dtype=torch.float32)
        self.transform = transform

    def __len__(self):

        return len(self.paths_dataset)

    def __getitem__(self, idx):

        jpg_image = Image.open(self.paths_dataset[idx])

        if self.transform:
            jpg_image = self.transform(jpg_image)

        return self.labels[idx], jpg_image

    def return_labels(self):
        return self.labels

class Tools():

    @staticmethod
    def generat_lists(path_to_folder: str, ratio: int):

        # function generate lists to train and valid with ratio
        # given as argument

        random.seed(42, version=2)
        paths_dataset = np.array(glob.glob(os.path.join(path_to_folder, '*/*.jpg')))
        n_total = len(paths_dataset)
        n_train = int(ratio * n_total)
        n_test = n_total - n_train
        train_idx = random.sample(range(0, len(paths_dataset)), k=n_train)
        test_idx = [i for i in range(0, len(paths_dataset)) if i not in train_idx]
        path_train_dataset = list(paths_dataset[train_idx])
        path_test_dataset = list(paths_dataset[test_idx])

        return path_train_dataset, path_test_dataset

    @staticmethod
    def set_seed(seed=42):

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark =False
        torch.manual_seed(seed)
        np.random.seed(seed)

    @staticmethod
    def save_checkpoint(path, epoch, model_state, optimizer_state, train_loss, valid_loss):

        # save for further learning

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(path_to_model: str,
                        model: torch.nn.Module,
                        optimizer: torch.optim.Adam,
                        train_loss: float,
                        valid_loss: float):

        # load model for further learning or inference

        checkpoint = torch.load(path_to_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_loss = train_loss
        valid_loss = valid_loss

        return model, optimizer, train_loss, valid_loss

    @staticmethod
    def predict(model: str,
                path_to_file: str):

        # use this function to inference

        #path_to_predict_model = 'exp/model_20_16_0.001_cuda_20240722_092927_8.pt'
        #path_to_image_to_predict = 'data/cats_or_dogs/Cat/200.jpg'
        model = CNNmodel3(input_shape=3, hidden_units=64, output_shape=1)
        checkpoint = torch.load(path_to_predict_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        jpg_image = Image.open(path_to_image_to_predict)
        transfom_composer = transforms.Compose([
            TransformChannels(),
            transforms.Resize(size=(100, 100)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        jpg_image = transfom_composer(jpg_image).unsqueeze(0)
        y_predict = model(jpg_image)
        result = 'Cat ' if y_predict.detach().cpu().numpy() < 0.5 else 'Dog'
        print(result, y_predict.detach().cpu().numpy())


    @staticmethod
    def accuracy(y_predicted: torch.Tensor, y_true: torch.Tensor):
        y_predicted_round = torch.where(y_predicted > 0.5, 1, 0)
        correct = torch.eq(y_predicted_round.squeeze(), y_true).sum().item()
        acc = (correct / len(y_predicted))
        return acc


class EarlyStopping:

    def __init__(self, patience=10):

        self.patience = patience
        self.counter = 0
        self.early_stopping = False

    def __call__(self, current_acc, best_acc):

        if current_acc > best_acc:
            best_acc = current_acc
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stopping = True
                return self.early_stopping

class CNNmodel2(nn.Module):

    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.block1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1,
                            padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            torch.nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 32 * 32, out_features=output_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)

        return x

class CNNmodel3(nn.Module):

    def __init__(self, input_shape=3, hidden_units=64, output_shape=1):
        super().__init__()
        self.block1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            torch.nn.Conv2d(in_channels=hidden_units, out_channels=512, kernel_size=3, padding=1),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.Dropout2d(0.2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.MaxPool2d(2) )

        self.classifier = nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=512 * 12 * 12, out_features=output_shape),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)

        return x

class CNNmodel4(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3))
        self.bn0 = nn.BatchNorm2d(num_features=16)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(num_features=32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(num_features=128)

        self.d = nn.Dropout(p=0.5)
        self.fc0 = nn.Linear(in_features=128*4*4, out_features=64)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):

        x = F.relu(self.bn0(self.conv0(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.d(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(F.relu(self.fc2(x)))

        return x

class TransformChannels(nn.Module):

    # some images are one channel encoded so
    # we need to convert them to RGB

    def forward(self, jpg_image):
        if jpg_image.mode != 'RGB':
            jpg_image = jpg_image.convert(mode='RGB')
        return jpg_image

if __name__ == "__main__":


    # parser = argparse.ArgumentParser(description='Recognizer objects on images')
    # parser.add_argument('--predict',
    #                     action='store_true',
    #                     default=False,
    #                     help='Use this option if you want only prediction apply')
    # parser.add_argument('--path_to_predict_model',
    #                     type=str,
    #                     help='Path to model. Use this only for prediction')
    # parser.add_argument('--path_to_image_to_predict',
    #                     type=str,
    #                     help='Path to image. Use this only for prediction')
    # parser.add_argument('--path_to_dataset',
    #                     type=str,
    #                     help='Path to database - one folder structure')
    # parser.add_argument('--n_epochs',
    #                     type=int,
    #                     help='Number of epochs')
    # parser.add_argument('--learning_rate',
    #                     type=float,
    #                     help='Learning rate')
    # parser.add_argument('--batch_size',
    #                     type=int,
    #                     help='Batch size')
    #
    # args = parser.parse_args()
    # eval = args.predict

    eval = False

    if eval:
        # path_to_predict_model = args.path_to_predict_model
        # path_to_image_to_predict = args.path_to_image_to_predict

        path_to_predict_model='exp/model_10_32_0.01_cuda_20240716_130308_4.pt'
        path_to_image_to_predict = 'data/cats_or_dogs/Cat/1.jpg'
        model = CNNmodel3(input_shape=3, hidden_units=64, output_shape=1)
        checkpoint = torch.load(path_to_predict_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        Tools.predict(model, path_to_image_to_predict)
        quit()

    # path_to_dataset = args.path_to_dataset
    # n_epochs = args.n_epochs
    # lr = args.learning_rate
    # batch_size = args.batch_size

    idx = 0
    dt = datetime.now().strftime('%Y%m%d_%H%M%S')
    n_epochs = 20
    batch_size = 4
    lr = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_tag = f'model_{n_epochs}_{batch_size}_{lr}_{device}_{dt}'
    models_path = os.path.join('exp', model_tag)
    writer = SummaryWriter(f'logs/{model_tag}')

    path_train_dataset, path_test_dataset = Tools.generat_lists('data/cats_or_dogs', 0.8)
    #path_train_dataset, path_test_dataset = Tools.generat_lists(path_to_dataset, 0.8)

    Tools.set_seed(seed=42)

    train_composer = transforms.Compose([
                              TransformChannels(),
                              transforms.Resize(size=(100, 100)),
                              transforms.RandomRotation(degrees=20),
                              transforms.RandomHorizontalFlip(p=0.5),
                              transforms.RandomVerticalFlip(p=0.005),
                              transforms.RandomGrayscale(p=0.5),
                              transforms.ToTensor(),
                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                              ])

    val_composer = transforms.Compose([
                                       TransformChannels(),
                                       transforms.Resize(size=(100, 100)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ])

    train_dataset_augm = CustomDataset(path_train_dataset, transform=train_composer)
    test_dataset = CustomDataset(path_test_dataset, transform=val_composer)

    train_loader = DataLoader(train_dataset_augm,
                              batch_size=32,
                              shuffle=True,
                              drop_last=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False)

    #model = CNNmodel3(input_shape=3, hidden_units=64, output_shape=1)
    model = CNNmodel4()
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()
    #loss_fn = torch.nn.BCEWithLogitsLoss()
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=6, verbose=True)
    # from pytorch doc:
    #   start_factor - the number we multiply lr in 1st epoch.
    #                  It changes towards end_factor in the following epochs
    #                  default 1/3
    # initialization of early stopping variabels
    early_stopping = EarlyStopping()

    model.to(device)

    for epoch in range(n_epochs):

        running_loss = 0
        valid_loss = 0
        num_total = 0
        num_total_valid = 0
        train_accuracy = 0
        valid_accuracy = 0
        acc_train = 0
        acc_valid = 0
        c = 0
        real_batch_size = len(train_loader)
        real_batch_size_valid = len(test_loader)


        for label, data in train_loader:
            model.train()

            batch_data = data.to(device)
            batch_label = label.to(device)

            y_predicted = model(batch_data)
            acc_train += Tools.accuracy(y_predicted=y_predicted, y_true=batch_label)
            l = loss_fn(y_predicted, batch_label.reshape(-1, 1))

            running_loss += l.item()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            c += 1

        scheduler.step()

        model.eval()
        with torch.inference_mode():
            for label_valid, data_valid in test_loader:

                batch_data_valid = data_valid.to(device)
                batch_label_valid = label_valid.to(device)

                y_predicted_valid = model(batch_data_valid)
                acc_valid += Tools.accuracy(y_predicted=y_predicted_valid, y_true=batch_label_valid)
                l_valid = loss_fn(y_predicted_valid, batch_label_valid.reshape(-1, 1))
                valid_loss += l_valid.item()

        running_loss /= real_batch_size
        valid_loss /= real_batch_size_valid

        train_accuracy = acc_train / real_batch_size
        valid_accuracy = acc_valid / real_batch_size_valid

        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('train loss', running_loss, epoch)
        writer.add_scalar('test loss', valid_loss, epoch)

        print(
            f'Epoch: {epoch} of {n_epochs}, loss on train {running_loss:.3f}, loss on valid {valid_loss:.3f}, accuracy on train {train_accuracy:.3f}, accuracy on valid {valid_accuracy:.3f}'
        )
        if (epoch % 4 == 0 and epoch != 0):
            print(f'Checkpoint at epoch {epoch} saved')
            models_path = os.path.join('exp', f'{model_tag}_{epoch}.pt')
            Tools.save_checkpoint(path=models_path, epoch=epoch,
                                  model_state=model.state_dict(),
                                  optimizer_state=optimizer.state_dict(),
                                  train_loss=running_loss,
                                  valid_loss=valid_loss)

        if early_stopping(train_accuracy, valid_accuracy):
            print(f'Early stopping at {epoch} epoch')
            break

    if not os.path.exists(models_path):
        models_path = os.path.join('exp', f'{model_tag}_end.pt')
        torch.save(model.state_dict(), models_path)








