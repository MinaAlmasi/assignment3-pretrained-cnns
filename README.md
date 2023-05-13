# Using Pretrained CNNs for Image Classification
Repository link: https://github.com/MinaAlmasi/assignment3-pretrained-cnns

This repository forms *assignment 3* by Mina Almasi (202005465) in the subject Visual Analytics, Cultural Data Science, F2023. The assignment description can be found [here](https://github.com/MinaAlmasi/assignment3-pretrained-cnns/blob/master/assignment-desc.md).

The repository contains code for training and evaluating a classifier using the pretrained CNN ```VGG16``` to supply image embeddings for the images. Please see the [*Results*](https://github.com/MinaAlmasi/assignment3-pretrained-cnns/blob/master/README.md#results) section for loss curves and evaluation metrics. 

## Data
The classification will be performed on the [*Indo Fashion dataset*](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset) (Rajput & Aneja, 2021). The dataset contains 106,000 images of Indian clothing seperated into 15 unique categories (examples include ```lehenga``` and ```palazzo```).

## Reproducibility
To reproduce the results, follow the instructions in the [*Pipeline*](https://github.com/MinaAlmasi/assignment3-pretrained-cnns#pipeline) section. The results display a model training on all the data, but instructions on how to run it on a subset of the data are also described. 

NB! The classification pipeline is computationally heavy. For this reason, cloud computing (e.g., UCloud) is encouraged.

## Project Structure
The repository is structured as such: 
```
├── README.md
├── assignment-desc.md
├── images                           <---    download and place data here !
│   └── README.md
├── results                          <---    results from model training & evaluation stored here ! 
│   ├── history_5_epochs.png
│   ├── metrics_5_epochs.txt
│   └── model_card.txt
├── requirements.txt
├── run.sh                           <---    to run classification pipeline !
├── setup.sh                         <---    to install necessary packages & reqs
└── src
    ├── classify_CNN.py              <---    classification pipeline to train & evaluate VGG16
    ├── classify_fns.py              <---    functions to intialise, optimise and train VGG16 
    ├── data_fns.py                  <---    functions to load load & preprocess data
    └── evaluate_fns.py              <---    functions to evaluate model (metrics, plots)
```

## Pipeline
The pipeline has been tested on Ubuntu v22.10, Python v3.10.7 ([UCloud](https://cloud.sdu.dk/), Coder Python 1.77.3). 
Python's [venv](https://docs.python.org/3/library/venv.html) needs to be installed for the pipeline to work.

### Setup
First, please download the [*Indo Fashion Kaggle dataset*](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset) and place all files in the ```images``` folder. Ensure that the data follows the structure and naming conventions described in [images/README.md](https://github.com/MinaAlmasi/assignment3-pretrained-cnns/tree/master/images).

Secondly, create a virtual environment (```env```) and install necessary requirements by running: 
```
bash setup.sh
```

### Running the Classification
To train and evaluate a classifier using```VGG16``` on the entire dataset for ```5``` epochs, run: 
```
bash run.sh
```

### Classification on a Subset of the Dataset
To run the classification pipeline on a **subset** of the dataset run ```classify_CNN.py``` with the additional arguments:
```
python src/classify_CNN.py -n_train 25000 -n_testval 5000 -epochs 5 
```



| Arg          | Description                         | Default       |
| :---         |:---                                 |:---           |
| ```-n_train```     | size of train data subset                 | None <br /> (i.e., all data) |
| ```-n_testval```    | size of test and validation data subset    | None (i.e., all data)            |
| ```-epochs```    | the amount of epochs that the model should run for  | 5          |


In the example above, the model will train on a subset of ```25000``` datapoints for ```5``` epochs with a both a test and validation set of ```5000``` datapoints.

NB! Remember to activate the ```env``` first (by running ```source ./env/bin/activate```)


## Results 
The sections below show the results from both model training and model evaluation. The model was trained for ```5``` epochs on the entire dataset. Model specifications can be found in [results/model_card.txt](https://github.com/MinaAlmasi/assignment3-pretrained-cnns/blob/master/results/model_card.txt). 

### Loss and Accuracy Curves 
<p align="left">
  <img src="https://github.com/MinaAlmasi/assignment3-pretrained-cnns/blob/master/results/history_5_epochs.png">
</p>

The loss curves for validation and training loss follow each other in the first four epochs before validation loss begins to rise in the fifth epoch. This is also reflected in the accuracy curves with the validation accuracy decreasing in the last epoch. This indicates that the model is underfitting to the validation data.

### Model Metrics
```
                      precision    recall  f1-score   support

              blouse       0.94      0.96      0.95       500
         dhoti_pants       0.85      0.61      0.71       500
            dupattas       0.70      0.75      0.73       500
               gowns       0.74      0.46      0.57       500
           kurta_men       0.78      0.90      0.84       500
leggings_and_salwars       0.65      0.83      0.73       500
             lehenga       0.91      0.88      0.90       500
         mojaris_men       0.89      0.76      0.82       500
       mojaris_women       0.80      0.90      0.85       500
       nehru_jackets       0.93      0.88      0.90       500
            palazzos       0.93      0.69      0.79       500
          petticoats       0.90      0.86      0.88       500
               saree       0.79      0.90      0.84       500
           sherwanis       0.91      0.81      0.85       500
         women_kurta       0.58      0.86      0.69       500

            accuracy                           0.80      7500
           macro avg       0.82      0.80      0.80      7500
        weighted avg       0.82      0.80      0.80      7500
```
The macro-averaged F1 is ```0.80``` for the model, but there are definitely individual variations in F1 score amongst classes. The model is especially bad at classifying ```gowns``` (```F1 = 0.57```) and ```women_kurta``` (```F1 = 0.69```) while being super good at the category ```blouse``` (```F1 = 0.95```). 

### Discussion on the Results
The loss curves indicate that the model training has not been entirely sucessful, but model performance is still quite high (```macro avg F1 = 0.80```) although there are considerable differences between classes in ```F1``` scores. For future work, the model training could be redone with more complex classification layers (higher hidden layer size than the current ```64``` and/or extra layers) and perhaps for more epochs.

## Author 
This repository was created by Mina Almasi:

* github user: @MinaAlmasi
* student no: 202005465, AUID: au675000
* mail: mina.almasi@post.au.dk
