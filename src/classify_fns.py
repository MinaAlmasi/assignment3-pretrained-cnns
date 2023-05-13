'''
Script for Assignment 3, Visual Analytics, Cultural Data Science, F2023

This script comprises several functions which make up a pipeline for training the CNN VGG16.

@MinaAlmasi
'''

# VGG16 model
from tensorflow.keras.applications.vgg16 import VGG16

# layers
from tensorflow.keras.layers import Flatten, Dense

#  generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

# functions
def intialize_vgg16_classifier(hidden_layer_size:int, output_layer_size:int): 
    '''
    Initialise VGG16 without its classifier layers in order to train a new simple neural network with VGG16's weights on a new classification task.

    Args: 
        - hidden_layer_size: size of the hidden_layer of the new classifier network added to VGG16
        - output_layer_size: size of the output layer. Should correspond to amount of unique classes to predict

    Returns: 
        - VGG16 with newly defined classifier layers. 

    '''

    # intialise model without classifier layers
    model = VGG16(include_top=False, 
              pooling='avg',
              input_shape=(224, 224, 3))

    # disable convolutional layers prior to model training
    for layer in model.layers:
        layer.trainable = False

    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(hidden_layer_size, activation='relu')(flat1)
    output = Dense(output_layer_size, activation='softmax')(class1) # output_layer_size should correspond to amount of unique classes to predict !  

    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)

    return model

def optimise_model(model): 
    '''
    Define a dynamic learning rate and compile the model with it (model optimisation).

    Args: 
        - model: intialised CNN model 
    
    Returns: 
        - model: model compiled with new learning rate ! 
    '''

    # define optimisation schedule 
    lr_schedule = ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9)
    
    # insert optimisation schedule in algorithm
    sgd = SGD(learning_rate=lr_schedule)

    # compile model with optimisation algorithm
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, train_data, validation_data, epochs:int):    
    '''
    Train initalised CNN for a specified amount of epochs. Evaluate model on validation data. 

    Args: 
        - model: intialised CNN
        - train_data: the data to be trained on 
        - validation_data: the data that the model evaluated on 
        - epochs: number of epochs that the model should train for

    Returns: 
        - history: history object containing information about the model training (e.g., loss and accuracy) 
            see documentation for more info: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History 
    '''
    # train model
    history = model.fit( # batch_size is not defined in model.fit as documentation specifies that is should not be done when using a data generator (https://www.tensorflow.org/api_docs/python/tf/keras/Model)
        train_data, 
        validation_data = validation_data,
        epochs=epochs, 
        verbose=1, 
        use_multiprocessing=True
        )

    return history