import numpy as np
import tensorflow as tf # backend for keras
import keras

def guesser(image):
    model = keras.models.load_model('number_reader.model')

    predictions = model.predict(image)
    answer = (np.argmax(predictions[0]))
    return answer


