import os
import numpy
from keras.models import model_from_json

class ModelSerializer:

    def __init__(self, model):
        self.model = model

    def serialize_model(model, model_filename, model_weights_filename):
        """
        This function saves a Machine Learning, Deep Learning or AI model
        so it may be reused quickly.
        This eliminates the need to re-train and re-compile to model

        model_filename: will have suffix .json, do not include in parameter
        model_weights_filename: will have suffix .h5, do not include in parameter
        """
        with open(model_filename + '.json', 'w') as json_file:
            #Serialize model to json
            json_file.write(model.to_json())
        # Serialize weights to HDF5
        model.save_weights(model_weights_filename +'.h5')

    def load_model(model_filename, model_weights_filename, loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics =['accuracy']):
        """
        Include suffix for each file in parameter
        """
        json_file = open(model_filename, 'r')
        model = json_file.read()
        json_file.close()
        model = model_from_json(model)
        model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
        return model
