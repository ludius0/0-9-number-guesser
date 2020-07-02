import numpy as np
from time import time

def train_network(Network, epochs=1):
    # load the mnist training data CSV file into a list
    training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    lenght = len(training_data_list)

    # Set up timer
    start = time()
    print("Start training")
    for e in range(1, epochs+1):
        for n, record in enumerate(training_data_list):
            all_records = record.split(',')
            # scale and shift the inputs
            inputs = (np.asfarray(all_records[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(Network.output_shape) + 0.01
            # Target value -> first in all_records (all_record[0])
            targets[int(all_records[0])] = 0.99
            
            Network.train(inputs, targets)
            
            if n % 5000 == 0 and n != 0:
                print(f"Already trained on {n} / {lenght} images. (Epochs: {e} / {epochs})")

        print(f"All {n+1} images trained in {e} epochs")
    print(f"Training was done in {round(time() - start, 2)} seconds")



def test_network(Network):
    # load the mnist test data CSV file into a list
    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    lenght = len(test_data_list)
    
    start = time()
    # how well the network performs
    scoreboard = []

    print("Testing network on unseen MNIST images.")
    # go through all the records in the test data set
    for n, record in enumerate(test_data_list):
        all_records = record.split(',')
        inputs = (np.asfarray(all_records[1:]) / 255.0 * 0.99) + 0.01
        targets = int(all_records[0])
        
        # query the network
        outputs = Network.query(inputs)
        # the index of the highest value corresponds to the label
        target_label = np.argmax(outputs)
        # If network's answer matches correct answer, add 1 to scorecard else append 0
        if (targets == target_label):
            scoreboard.append(1)
        else:
            scoreboard.append(0)
        if n % 1000 == 0 and n != 0:
            print(f"Already tested on {n} / {lenght} images.")

    print(f"Testing is done on overall {n+1} images. It took {round(time() - start, 2)} seconds")
    # calculate the performance score, the fraction of correct answers
    scoreboard_array = np.asarray(scoreboard)
    print (f"Accuracy: {(scoreboard_array.sum() / scoreboard_array.size) * 100}%")
