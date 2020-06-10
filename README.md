# 0-9-number-guesser

**Number Gusser of 0 to 9**

As I was learning and playing with deep learning (and machine learning) I wanted to test my self and make something. In keras is mnist dataset of already labeled pictures of numbers (28x28 size). I used it to create model and rest is only GUI. \
Controls: \
Mouse: \
  left click -> draw \
  right click -> erase \
 Keyboard: \
  space -> load model and guess number (than erase whole display) \

![ezgif com-gif-maker(6)](https://user-images.githubusercontent.com/57571014/84257428-99219800-ab15-11ea-96d5-ebeb253d4b83.gif)\n

In background is board of (1, 28, 28, 1) dimensions, it is list of pixels of your picture you draw (on grid of 28x28 rects/pixels).
There is one added script "create_model.py". If there isn't model in directory with "main.py" or if you want tweak model you can  used that script.

The model isn't best. I found that it straggle if you draw around center and not directly in center; also if you draw big numbers. But overall it work and I'm satisfied with result. 
BTW: The model took me about 30 minutes to train. (GPU: GTX 960; at least I think it used GPU)
