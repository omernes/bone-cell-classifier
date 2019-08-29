# Bone Cell Classifier

## Introduction
This project trains an SSD Object Detection model to identify and count Osteoclast bone cells in microscopic images.
In here are scripts to create a dataset from a pre-defined JSON format, train and evaluate a model, and count cells in an image based on that model.


## Creating the Dataset
The script 'create_dataset.py' will iterate over the JSON files supplied, and for each image in the original data, the script will create sub-images with a size of (300,300) in a sliding windows format with steps of 100 pixels (all values are configurable inside the script). 
It will create two directories:
* Images - containing the PNG files of the images.
* Annotations - containing XML files in PASCAL VOC format of the polygons in each image.
After, it will split the new dataset into 'train', 'val' and 'test' parts by creating three TXT files, each corresponding to a different stage which contains the name of the images that belong to it.

For easier control over the envrironment variables that define the paths to the data, they're all included in 'run_create_dataset.sh'.

## Training the Model
The model is being trained based on the PASCAL VOC model available online. Running 'run_subsample_weights.sh' will take that model and convert it from 20 categories structure into 5 (which is our case).

To train the model, use 'run_finetuning.sh', and set the environment variables as necessary. It's possible to control number of epochs, learning rate, batch size, optimizer, etc. The model will be saved every X epochs (defined in an environment variable) to a directory specified, if it has improved its loss value.
The training script expects paths to the images and annotations directories, as well as the text files that define the train, val and test datasets. 


## Predicting and Evaluating
To evaluate a model, use the 'run_evaluate.sh' script, that sets the MODEL_WEIGHTS_PATH environment variable that points to the extracted weights from a model.
In order to extract the weights, use 'run_extract_weights.sh' on a trained model.

The evaluation process is simple: predicting all of the images in the test dataset, and comparing to actual results using IoU for each cell identified. Results are given for each category and a total value for the model.


## Processing an Image
The script 'process_image.py' receives a path to an image as an argument; it then splits the images to windows using the same method as in creating the dataset, and predicts each part of the image on its own.
After getting all of the predictions, it eliminates overlapping identified cells and keeps only one that are standalone. The script saves the coordinates of the cells in a JSON file along with the counters (for every cell type), and also saves a copy of the image with the cells tagged.
Both JSON and PNG files are saved to the dirctory the script is in.