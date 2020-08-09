#  参考にしたQiita　https://qiita.com/hiraku00/items/66a3606af3b2eed57778

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from PIL import Image
import sys


if __name__ == '__main__':
    
    classes = ['turtle', 'moon']
    num_classes = len(classes)
    IMAGE_SIZE = 224

    X = []

    # convert data by specifying file from terminal
    image = Image.open('./Turtle_Moon/000.jpg')
    image = image.convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    data = np.asarray(image)
    X.append(data)
    X = np.array(X)
    X = X.astype('float') / 255.0

    # load model
    model = load_model('./Turtle_Moon/vgg16_transfer_turtle_moon_classes.h5')

    # estimated result of the first data (multiple scores will be returned)
    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)

    print(classes[predicted], percentage)
