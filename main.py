import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

class MnistDataset(Dataset):
    def __init__(self, path):
        # CSV reader
        training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
        self.data_df = training_data_file.readlines()
        training_data_file.close()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        img = list(map(int, self.data_df[index].split(',')))
        label = img[0]
        target = torch.zeros((10))
        target[label] = 1.0
        img_values = torch.FloatTensor(img[1:]) / 255.0

        return label, img_values, target

    def plot(self, index):
        img = list(map(int, self.data_df[index].split(',')))
        arr = np.array(img[1:]).reshape(28, 28)
        plt.title("label = " + str(img[0]))
        plt.imshow(arr, interpolation="none", cmap="Blues")
        plt.show()



class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.Sigmoid(),
            nn.Linear(200, 10),
            nn.Sigmoid()
            )
        self.loss_function = nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        # Timestap of loss for ploting progress
        self.progress = []
        self.counter = 0

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)

        loss = self.loss_function(outputs, targets)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.counter += 1
        self.progress.append(loss.item())
        
        if self.counter % 10000 == 0:
            print(f"Already trained on {self.counter} images.")
    
    def plot_progress(self):
        x_size = [i for i in range(self.counter)]

        # Plot
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_title("Loss over time (x=trained images, y=loss values)")
        ax.plot(x_size, self.progress)
        plt.show()

path = "mnist_dataset/mnist_train.cvs"

dataset = MnistDataset(path)
classifier = DigitClassifier()

#print(len(dataset))
#dataset.plot(0)

### TRAIN ###
epochs = 1

for e in range(1, epochs+1):
    print(f"Training in {e} / {epochs} epochs.")
    for idx, (label, img_data_tensor, target_tensor) in enumerate(dataset):
        classifier.train(img_data_tensor, target_tensor)

        if idx == 10000:
            break

classifier.plot_progress()
classifier.counter = 0

print("Code runned!")