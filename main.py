import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

class MnistDataset(Dataset):
    def __init__(self, path):
        """
        Load data in class
        """
        # CSV reader
        training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
        self.data_df = training_data_file.readlines()
        training_data_file.close()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        """
        Get from 'for loop' (label, tensor raw image, target of image (what should NN output))
        """
        img = list(map(int, self.data_df[index].split(',')))
        label = img[0]
        target = torch.zeros((10))
        target[label] = 1.0
        img_values = torch.FloatTensor(img[1:]) / 255.0

        return label, img_values, target

    def plot(self, index):
        """
        Plot (show) targeted (index) image
        """
        img = list(map(int, self.data_df[index].split(',')))
        arr = np.array(img[1:]).reshape(28, 28)
        plt.title("label = " + str(img[0]))
        plt.imshow(arr, interpolation="none", cmap="Blues")
        plt.show()



class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Create structure of NN
        """

        # Neural Network layers
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 10),
            nn.Sigmoid()
            )
        # Loss function; tweaking weights in backpropagation; BCELoss need input between 0 to 1 (NN output layer have to be 0 to 1 (Sigmoid or Normalised layer))
        self.loss_function = nn.BCELoss()
        # Adam optimiser give "rolling ball" momentum in gradiant descent
        self.optimiser = torch.optim.Adam(self.parameters())

        # Timestap of loss for ploting progress
        self.progress = []
        self.counter = 0

    def forward(self, inputs):
        """
        Pass through NN and get its answer
        """
        return self.model(inputs)

    def train(self, inputs, targets):
        """
        Train NN; Take tensor of image with label identificator of image;
        Pass through NN; get loss/cost function and backpropagate NN
        to tweak weights (layers)
        """
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)

        # Backpropagation
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.counter += 1
        self.progress.append(loss.item())

        if self.counter % 10000 == 0:
            print(f"Already trained on {self.counter} images.")

    def plot_progress(self):
        """
        Plot loss of NN for every image it was trained
        """
        x_size = [i for i in range(self.counter)]

        # Plot
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_title("Loss over time (x=trained images, y=loss values)")
        ax.plot(x_size, self.progress)
        plt.show()

path = "mnist_dataset/mnist_train.cvs"

train_dataset = MnistDataset(path)
classifier = DigitClassifier()

#print(len(dataset))
#dataset.plot(0)

### TRAIN ###
epochs = 3
for e in range(1, epochs+1):
    print(f"Training in {e} / {epochs} epochs.")
    for idx, (label, img_data_tensor, target_tensor) in enumerate(train_dataset):
        classifier.train(img_data_tensor, target_tensor)


classifier.plot_progress()
classifier.counter = 0


### TEST ###

path = "mnist_dataset/mnist_test.cvs"

test_dataset = MnistDataset(path)

items, score = 0, 0

print("Testing accuracy")
for idx, (label, img_data_tensor, target_tensor) in enumerate(test_dataset):
    answer = classifier.forward(img_data_tensor).detach().numpy()
    if answer.argmax() == label:
        score += 1
    items += 1

print(f"Accuracy: {score/items*100} %")

print("Code runned!")
