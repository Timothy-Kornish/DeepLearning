#-------------------------------------------------------------------------------
#		                 Convolution Neural Network
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#		                 Libraries
#-------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import os
import sys
sys.path.append('../')
from util import ModelSerializer


#-------------------------------------------------------------------------------
#		                 Initializing the CNN
#-------------------------------------------------------------------------------

classifier = Sequential()

#-------------------------------------------------------------------------------
#		                 Step 1 - Convolution
#-------------------------------------------------------------------------------
"""
Convolution2D(nb_filter=64, 3, 3, border_mode='same', input_shape=(3, 256, 256))

64 featue detectors of 3 by 3 dimesnions creating 64 feature maps for step 2

border_mode is how the detector will treat the border features of the input image
compared to other features

input_shape = expected format of input images of RGB (3, 256, 256) for 256 * 256 RGB pictures
"""
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu"))

#-------------------------------------------------------------------------------
#		                 Step 2 - Max Pooling
#-------------------------------------------------------------------------------
"""
Max poolion takes strides by two and pulls the
largest value of the 2 by 2 matrix into a new reduce size image
"""
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#-------------------------------------------------------------------------------
#		                 Adding a second Convolutional layer, input_shape is not required a second time
#-------------------------------------------------------------------------------

classifier.add(Convolution2D(32, (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#-------------------------------------------------------------------------------
#		                 Step 3 - FLattening
#-------------------------------------------------------------------------------
"""
Flatten all pictures from pooling into a single vector
"""
classifier.add(Flatten())

#-------------------------------------------------------------------------------
#		                 Step 4 - Full Connection
#-------------------------------------------------------------------------------

#classifier.add(Dense(units = 1, activation = "relu"))
classifier.add(Dense(units = 1, activation = "sigmoid"))

#-------------------------------------------------------------------------------
#		                 Compile the CNN
#-------------------------------------------------------------------------------

classifier.compile(optimizer = "adam", loss = "binary_crossentropy",
                   metrics = ["accuracy"])

#-------------------------------------------------------------------------------
#		                 Fitting the CNN to the images
#-------------------------------------------------------------------------------

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')

test_set= test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')
# typically steps_per_epoch is divided by batch_size but not necessary
# typically validation_steps is divided by batch_size but not necessary
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/32,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000/32)

#-------------------------------------------------------------------------------
#		                 Serializing CNN Model
#-------------------------------------------------------------------------------

ModelSerializer.serialize_model(classifier, 'model', 'weights')

#-------------------------------------------------------------------------------
#		                 Making predictions on single images
#-------------------------------------------------------------------------------


test_image_1 = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image_2 = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))

test_image_1 = image.img_to_array(test_image_1)
test_image_2 = image.img_to_array(test_image_2)

# adding a 4th dimension for predict method
# this dimension corresponds to the batch, cannot accept single inputs
# only batch of size single or greater inputs
test_image_1 = np.expand_dims(test_image_1, axis = 0)
test_image_2 = np.expand_dims(test_image_2, axis = 0)

predict_1 = classifier.predict(test_image_1)
predict_2 = classifier.predict(test_image_2)

# to understand output of 0 or 1, cats are 0, dogs are 1
training_set.class_indices

if predict_1[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print("The image 1 is a ", prediction)

if predict_2[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print("The image 2 is a ", prediction)
