'''
Script for Assignment 3, Visual Analytics, Cultural Data Science, F2023

The following script contains functions to load a dataset and preprocess the Indo Fashion dataset using Tensorflow. 

The dataset must follow this exact structure:

└── images
    ├── metadata      <--- JSON files containing metadata for each set of data with the naming convention: test_data.json, train_data.json, val_data.json
    ├── test          <--- JPEG files as test set
    ├── train         <--- JPEG files as train set
    └── val           <--- JPEG files as validation set

The images are resized to (224, 224) and preprocessed in the same way as VGG16's original training data (see paper here: https://arxiv.org/abs/1409.1556)

@MinaAlmasi
'''

# utils 
import pathlib

# load metadata
import pandas as pd
import re 

# import image dataset 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# image processsing
from tensorflow.keras.applications.vgg16 import preprocess_input # prepare image for VGG16

# functions
def load_metadata(metadatapath:pathlib.Path()):
    '''
    Loads metadata stored as JSON files in "metadatapath" as pandas dataframes in a dictionary. 

    Args: 
        - metadatapath: path where the JSON files are stored. 
            NB! The JSON files are required to have the following naming convention: "X_data.json" (e.g train_data.json) 

    Returns: 
        metadata: dictionary with each json file being a seperate pandas dataframe. 
            Follows the naming convention of the JSON files, but removing the _data.json extension (e.g., metadata["train"]).
    '''

    metadata = {}

    # iterate over paths in metadatapath 
    for file in metadatapath.iterdir(): 
        
        # from file.name (e.g., "test_data.json"), rm ".json"
        file_name = re.sub("_data.json", "", file.name) 

        # from file (filepath) read as pandas dataframe, call it the file_name and append to dfs dict (e.g., metadata["train"])
        metadata[file_name] = pd.read_json(file, lines=True) 

    return metadata

def load_tf_data(metadata_train, metadata_test, metadata_val, imagepath_col:str, label_col:str, image_size:tuple, batch_size:int):
    '''
    Loads and preprocesses training, test and validation data using Tensorflow's ImageDataGenerator and its method .flow_from_dataframe(). 

    Args: 
        - metadata_train, metadata_test, metadata_val: dataframes containing image paths (imagepath_col) and class labels (label_col)
        - imagepath_col: column containing image paths in the three dataframes (needs to be the same for all metadata!)
        - label_col: column containing class labels in the three dataframes (needs to be the same for all metadata!)
        - image_size: size that images should be resized to e.g., (224, 224) for VGG16
        - batch_size: size of batches that the data should be loaded in. Will be the same for all data.  

    Returns: 
        - train_data, test_data, val_data: data for both training, test and validation ready to be used with a tensorflow model 
    '''

    # intialise image data generator 
    datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input # preprocess using VGG16's function 
    )

    # load train
    train_data = datagen.flow_from_dataframe(
        dataframe = metadata_train, 
        x_col = imagepath_col, 
        y_col = label_col, 
        target_size = image_size, 
        batch_size = batch_size, 
        class_mode = "categorical", 
        shuffle = True, # shuffle data as the data is structured with all of the same class following each other 
        seed = 129
    )

    # load test
    test_data = datagen.flow_from_dataframe(
        dataframe = metadata_test,
        x_col = imagepath_col, 
        y_col = label_col, 
        target_size= image_size, 
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = False, 
        seed = 129
    )

    # load val
    val_data = datagen.flow_from_dataframe(
        dataframe = metadata_val, 
        x_col = imagepath_col, 
        y_col = label_col, 
        target_size= image_size, 
        batch_size = batch_size,
        class_mode = "categorical", 
        shuffle = True, 
        seed = 129
    )

    return train_data, test_data, val_data