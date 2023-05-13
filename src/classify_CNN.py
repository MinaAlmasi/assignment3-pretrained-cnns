'''
Script for Assignment 3, Visual Analytics, Cultural Data Science, F2023

This script contains a full classification pipeline for training and evaluating VGG16 on the Indo Fashion Kaggle Dataset (https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset).
The pipeline relies on functions defined in data_fns.py, classify_fns.py and evaluate_fns.py

To run the script type the following in the terminal: 
    python src/classify_CNN.py -n_train {NUMBER_OF_SAMPLES_TRAIN} -n_testval {NUMBER_OF_SAMPLES_TEST_VALIDATION} -epochs {NUMBER_OF_EPOCHS}

With additional arguments  
    -n_train: number of samples in train data
    -n_testval: number of samples in the test and validation data
    -epochs: number of epochs that the training is run for 

@MinaAlmasi
'''

# utils 
import pathlib
import argparse

# custom modules
from data_fns import load_metadata, load_tf_data
from classify_fns import intialize_vgg16_classifier, optimise_model, train_model
from evaluate_fns import save_model_card, plot_model_history, get_model_metrics, save_model_metrics

def input_parse():
    parser = argparse.ArgumentParser()

    # add arguments 
    parser.add_argument("-n_train", "--n_train_samples", help = "n samples in train_data", type = int, default=None)
    parser.add_argument("-n_testval", "--n_testval_samples", help = "n samples in val and train data", type = int, default=None)
    parser.add_argument("-epochs", "--n_epochs", help = "number of epochs the model is run for", type = int, default=5)

    # save arguments to be parsed from the CLI
    args = parser.parse_args()

    return args

def main():
    # define paths 
    path = pathlib.Path(__file__)
    metadatapath = path.parents[1] / "images" / "metadata"
    outpath = path.parents[1] / "results"

    # check if outpath is created
    outpath.mkdir(exist_ok=True) # if it already exists, continue! 

    # define args
    args = input_parse()

    # load metadata
    meta_dict = load_metadata(metadatapath)

    # sample meta_data (due to computational issues, it is at times difficult to run with the entire dataset hence the need for sampling !)
    if args.n_train_samples != None: # if args.n_samples is not None, then subset data:
        meta_dict["train"] = meta_dict["train"].sample(args.n_train_samples, random_state=10)
    if args.n_testval_samples != None:
        meta_dict["test"] = meta_dict["test"].sample(args.n_testval_samples, random_state=10)
        meta_dict["val"] = meta_dict["val"].sample(args.n_testval_samples, random_state=10)
    
    # extract train, test, val 
    train, test, val = meta_dict["train"], meta_dict["test"], meta_dict["val"] 

    # load data with tensorflow
    print("[INFO]: Loading data")
    train_data, test_data, val_data = load_tf_data(
        metadata_train = train, 
        metadata_test = test, 
        metadata_val = val,  
        imagepath_col= "image_path", 
        label_col = "class_label",
        image_size = (224, 224), # image size (224, 224) required for VGG16
        batch_size = 64
        ) 

    # intialize model
    print("[INFO]: Intializing model")
    model = intialize_vgg16_classifier(hidden_layer_size = 64, output_layer_size = 15)

    # optimise model
    model = optimise_model(model)

    # define epochs 
    n_epochs = args.n_epochs

    # train model
    print("[INFO]: Training model")
    model_history = train_model(model, train_data, val_data, n_epochs)

    # save model 
    modelpath = path.parents[1] / "models"  # define folder
    modelpath.mkdir(exist_ok=True) # make if it does not exist

    # save model 
    model.save(modelpath / f"model_{n_epochs}_epochs.h5")

    # saving model information
    save_model_card(model, n_epochs, outpath, "model_card.txt")

    # save plot history (training and validation loss)
    plot_model_history(model_history, n_epochs, outpath, f"history_{n_epochs}_epochs.png")

    # evaluate model 
    print("[INFO]: Evaluating model")
    metrics = get_model_metrics(model, test_data)

    # save metrics
    print("[INFO]: Saving model")
    save_model_metrics(metrics, outpath, f"metrics_{n_epochs}_epochs.txt")


if __name__ == "__main__":
    main()