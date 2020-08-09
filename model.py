import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD



PATH = './Turtle_Moon/'

num_classes = 2
IMAGE_SIZE = 224 # Specified size of VGG16 Default input size in VGG16


X_train, X_test, y_train, y_test = np.load(PATH + 'image_files_turtle_moon_classes.npy', allow_pickle=True)

# convert one-hot vector
# y_train = np_utils.to_categorical(y_train, num_classes)
# y_test = np_utils.to_categorical(y_test, num_classes)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# normalization
# X_train = X_train.astype('float') / 255.0
# X_test = X_test.astype('float') / 255.0
X_train = np.array(X_train, dtype=np.float32) / 255.0
X_test = np.array(X_test, dtype=np.float32) / 255.0

vgg16_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)   
)

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

# combine models
model = Model(
    inputs=vgg16_model.input,
    outputs=top_model(vgg16_model.output)
)
model.summary()

for layer in model.layers[:15]:
    layer.trainable = False

opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10)

score = model.evaluate(X_test, y_test, batch_size=32)
print('loss: {0} - acc: {1}'.format(score[0], score[1]))

model.save(PATH + 'vgg16_transfer_turtle_moon_classes.h5')


