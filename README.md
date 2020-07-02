# 0-9-number-guesser

**Neural Network Guessing Numbers**
----------------------------
**Third-party modules:**
- Numpy
- pillow

You need dataset from here: https://github.com/pjreddie/mnist-csv-png


![ezgif com-video-to-gif](https://user-images.githubusercontent.com/57571014/86401674-f0490180-bcaa-11ea-9f4b-d39c83d36b51.gif)

**INFO:**\
After you download MNIST dataset in csv format in mnist_dataset folder (Both: training set & testing set). 
You just start "main.py" script and neural network will start train it self. 
After that it will test it's accuracy on testing set and after that it will initialize tkinter GUI and open you canvas.
On canvas you draw digits and after that you just need to hit query and Neural network will give guess.

In "main.py" you play with settings of neural network -> NeuralNetwork(784, 56, 10, learning_rate=0.1) -> 
First number is input of image (MNIST image is 28*28=784); 56 is number of hidden neurons (change as you want); 10 is final output (0-9 digits); learning_rate is part of cost function
in "neural_network.py". If you want checkout how it works or play with settings, than go to "neural_network.py".


**NOTE:**\
Goal of this remake was create neural network without third-party frame-work and try it on my own. It wasn't easy, because there is big possibilities how to build neural network. 
And I had to learn concepts as grant-descent (and slope) & cost functions (+ headaches with numpy shapes of arrays), but it is done and I am happy and proud I was able to build it. It really helped me with low-level understanding of neural networks, because before with frameworks it was't so obvious and I wasn't able to do with them and ended up confused.
Also my cousin recommended to learn git, so I also used with developing this Neural Network and updated this repo.

Also it's pretty interesting as each time, when I start this neural network, which pattents will pick up. Also I recommend to draw bigger digits or draw similiar like form mnist dataset. Image get shrink (28x28) and convert it to numpy array.
