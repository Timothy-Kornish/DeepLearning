from tensorflow.contrib.keras.api.keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.contrib.keras.api.keras.models import Sequential, load_model
from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.api.keras.preprocessing import image
from tensorflow.contrib.keras import backend
import numpy as np
import os
import sys
sys.path.append('../')
from util import ModelSerializer

class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''

    def on_epoch_end(self, epoch, logs = {}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1

    def on_train_begin(self, logs = {}):
        self.losses += 'training begins...\n'

script_directory = os.getcwd()
train_set_path = script_directory + '/dataset/training_set'
test_set_path = script_directory + '/dataset/test_set'

# Part 1 Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
input_size = (128, 128)
classifier.add(Conv2D(32, (3, 3), input_shape = (*input_size, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2))) # 2 x 2 is optimal

# Add a second Convolution and mapping layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Add a third Convolution and mapping layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - FLattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Step 5 - Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

batch_size = 32
train_datagen = ImageDataGenerator(rescale =1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./ 255)

training_set = train_datagen.flow_from_directory(train_set_path,
                                                 target_size = input_size,
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size = input_size,
                                            batch_size = batch_size,
                                            class_mode = 'binary')

# Create a loss history
history = LossHistory()

# train model
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/batch_size,
                         epochs = 90,
                         validation_data = test_set,
                         validation_steps = 2000/batch_size,
                         workers = 12,
                         max_q_size = 100,
                         callbacks = [history])

# Serialize Model
ModelSerializer.serialize_model_json(classifier, 'loss_history', 'loss_history_weights')

# Predict single cases
test_image_1 = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = input_size)
test_image_2 = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = input_size)

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


# Save model
model_backup_path = script_directory + '/cats_or_dogs_model.h5'
classifier.save(model_backup_path)
print('Model saved to: ', model_backup_path)

# Save loss history
loss_history_path = script_directory + '/loss_history.log'
with open(loss_history_path, 'w') as myFile:
    myFile.write(history.losses)

backend.clear_session()
print('The model class indices are: ', training_set.class_indices)
