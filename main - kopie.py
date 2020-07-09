import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import mtplotlib.pyplot as plt

class MnistDataset(Dataset):
    def __init__(self, path):
        # CSV reader
        self.data_df = pd.read_csv(path, header=None)

    def plot(self, index):
        arr = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title("label = " + str(self.data_df.iloc[index, 0]))
        plt.imshow(arr, interpolation="none", cmap="Blues")

    def __len__(self):
        return self.data_df

    def __getitem__(self, index):
        label = self.data_df.iloc[index, 0]
        target = torch.zeros((10))
        target[label] = 1.0
        img_values = torch.FloatTensor(self.data_df.iloc[index, 1:].values) / 255.0

        return label, img_values, target

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200)
            nn.Sigmoid(),
            nn.Linear(200, 10)
            nn.Sigmoid()
            )
        self.loss_function = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)

        loss = self.loss_function(outputs, targets)

        self.optimiser.zero_grad()
        loss.backwards()
        self.optimiser.step()
