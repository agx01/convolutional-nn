# -*- coding: utf-8 -*-
#Check if tensorflow is using GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#Class for getting the accuracy and loss at end of epoch
import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()


#Building the CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Convolution Layer
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation = 'relu'))
#input shape is in tensorflow backend : channel is last parameter vice versa for theano

#Pooling Layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding another Convolution layer and Pooling Layer to increase accuracy
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening Layer
classifier.add(Flatten())

#Classic ANN or Fully Connected Layer
#hidden layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))

#output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Building the CNN end

#Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'Convolutional_Neural_Networks/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

testing_set = test_datagen.flow_from_directory(
        'Convolutional_Neural_Networks/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=2000,
        epochs=25,
        validation_data=testing_set,
        validation_steps=2000,
        callbacks=[plot_losses])



#Fitting the CNN to the images end