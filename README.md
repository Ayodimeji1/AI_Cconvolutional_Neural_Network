
# Convolutional Neural Network (CNN) Model for Image Classification

## Overview
This project demonstrates a Convolutional Neural Network (CNN) built using TensorFlow and Keras for image classification tasks. The primary focus of the model is to classify images into different categories using a deep learning approach. The project is developed using Python and includes essential exploration, dataset preparation, model training, and evaluation steps.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training and Evaluation](#training-and-evaluation)
6. [Results](#results)
7. [Usage](#usage)
10. [License](#license)

## Installation
Ensure that Python is installed on your system. Follow these steps to set up the environment:
1. Clone this repository.
 ```
git clone https://github.com/Ayodimeji1/CNN.git
2. Install the necessary dependencies:
 

## Dataset
The dataset used in this project consists of images of flowers, split into five categories. The data is loaded using TensorFlow's `tf.keras.utils.image_dataset_from_directory` method.

- **Number of images**: 3670
- **Number of categories**: 5
- **Image size**: Resized to 256x256 pixels

## Model Architecture
The CNN is built using TensorFlow's Keras API and consists of multiple convolutional and pooling layers followed by dense layers for classification. The key components include:

- **Convolutional Layers**: Extract spatial features.
- **Pooling Layers**: Reduce dimensionality.
- **Dense Layers**: Perform the final classification.

## Training and Evaluation
The model is trained on the dataset with a specified batch size and uses a validation split to monitor performance. The notebook contains details about the training configurations and metrics used for evaluation.

## Results
Details about the model's performance, including accuracy and loss plots, are shown in the notebook. 

## Usage
To use this model for your own dataset:
1. Ensure your images are organized in a directory structure similar to `dataset_name/class_name/image.jpg`.
2. Adjust the dataset path in the code:
   ```
   python
   dataset = tf.keras.utils.image_dataset_from_directory(
       'your_dataset_path', batch_size=500,
       image_size=(256, 256))
   ```
3. Run the training cells in the provided notebook.

## Dependencies
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib


## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
