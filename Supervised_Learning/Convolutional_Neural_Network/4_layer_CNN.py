from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from ketas.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 150, 150

def initialize_model(p = 0.5, input_shape = (32, 32, 3)):

    """
    Create a CNN model
    p: Dropout rate
    input_shape: shape of input
    """

    # initialize CNN
    model = Sequntial()

    # 4 layers of Convolution and pooling

    # 1st layer
    model.add(Convolution2D(32, (3, 3), padding = 'same', input_shape = input_shape, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # 2nd layer
    model.add(Convolution2D(32, (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # 3rd layer
    model.add(Convolution2D(32, (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # 4th layer
    model.add(Convolution2D(32, (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Flattening
    model.add(Flatten())

    # Full Connection
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(p))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(p/2))
    model.add(Dense(1), activation = 'sigmoid'))

    # Compiling CNN
    optimizer = Adam(lr = 1e-3)
    metrics = ['accuracy']
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = metrics)
    return model

def train_model(batch_size = 32, epochs = 20, img_width = img_width, img_height = img_height):

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size = (img_width, img_height),
                                                     batch_size = batch_size,
                                                     class_mode = 'binary')

    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size = (img_width, img_height),
                                                batch_size = batch_size,
                                                class_mode = 'binary')

    model = create_model(p = 0.6, input_shape = (img_width, img_height, 3))
    model.git_generator(training_set,
                        steps_per_epoch = 8000/batch_size,
                        epochs = epochs,
                        validation_data = test_set,
                        validation_steps = 2000/batch_size)

def main():
    train_model(batch_size = 32, epochs = 100)

if __name__ == '__main':
    main()
