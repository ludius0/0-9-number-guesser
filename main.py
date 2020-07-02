"""
Programmed by ludius0
"""
# Modules
import tkinter as tk
import numpy as np
from PIL import Image, ImageGrab

# Scripts
from neural_network import NeuralNetwork
import dataset_f as df

# Images from MNIST dataset are 28*28=784; We load them in vector array
# Hidden layer will have 100 neuron and output is 10 (0-9 digits)
output_nodes = 10
DigitGuesser = NeuralNetwork(784, 58, output_nodes, learning_rate=0.1)

# Train Neural Network on MNIST dataset of hand writed digits
df.train_network(DigitGuesser, epochs=5)
# Test Neural Network on MNIST that network never seen before and calculate accuracy
df.test_network(DigitGuesser)

if __name__ == "__main__":
    ### GUI ###

    class Painter():
        def __init__(self):
            self.root = tk.Tk()
            self.root.title("Digit Guesser by ludius0")

            self.width = 500
            self.height = 500
            self.color = "black"

            self.canvas1 = tk.Canvas(self.root, width=self.width, height=self.height, bg="white")
            self.canvas1.pack(expand=tk.NO, fill=tk.BOTH)
            self.canvas1.bind("<B1-Motion>", self.paint)

            self.button1 = tk.Button(self.root, text="Erase", command=self.erase)
            self.button1.pack(side="left", expand=True, ipadx=10, ipady=10)

            self.button2 = tk.Button(self.root, text="Guess", command=self.query)
            self.button2.pack(side="right", expand=True, ipadx=10, ipady=10)
            
            self.var = tk.StringVar()
            self.var.set(" ")
            self.label = tk.Label(self.root, textvariable=self.var)
            self.label.pack(side="bottom")

            self.root.mainloop()
            
        def paint(self, event):
            x1, y1 = (event.x-10), (event.y-10)
            x2, y2 = (event.x+10), (event.y+10)
            self.canvas1.create_oval(x1, y1, x2, y2, fill=self.color, outline=self.color)

        def erase(self):
            self.canvas1.delete("all")
            self.var.set(" ")
            self.root.update_idletasks()

        def use_network(self, img):
            inputs = (np.asfarray(img) / 255.0 * 0.99) + 0.01
            output = DigitGuesser.query(inputs).reshape(10,)
            output_n = np.argmax(output).tolist()
            self.output_n = str(output_n)
                
            self.var.set(self.output_n)
            self.root.update_idletasks()

        def query(self):
            # Get parameters
            x = self.root.winfo_rootx() + self.canvas1.winfo_x()
            y = self.root.winfo_rooty() + self.canvas1.winfo_y()
            x1 = x + self.canvas1.winfo_width()
            y1 = y + self.canvas1.winfo_height()

            # Grab image and convert it to black and white
            image = ImageGrab.grab().crop((x,y,x1,y1))
            gray = image.convert('L')
            
            # Use numpy
            bw = np.asarray(gray).copy()
            #bw[bw < 128] = 0    # Black
            #bw[bw >= 128] = 255 # White
            # Invert arary, because MNIST images are opposite colors
            bw = np.invert(bw)
            # Put it back in Pillow/PIL format
            main_image = Image.fromarray(bw)
            # Resize it to image like from MNIST dataset
            img = main_image.resize((28, 28))
            img_array = np.array(img).reshape(784,)

            self.use_network(img_array)

    app = Painter()

