# imageclassification# Image Classification using Convolutional Neural Networks

## Overview

This project demonstrates image classification using Convolutional Neural Networks (CNNs) implemented in TensorFlow and Keras. The model is trained to classify images of cats and dogs. The dataset consists of training and validation sets, with data augmentation applied during training to enhance model generalization.

## Project Structure

- **basedata/training**: Contains training images divided into subdirectories for cats and dogs.
- **basedata/validation**: Holds validation images for model evaluation.
- **basedata/testing**: Directory for testing the trained model on new images.

## Requirements

- TensorFlow
- Matplotlib
- OpenCV
- Numpy

## Data Preprocessing

- Images are loaded and preprocessed using the `ImageDataGenerator` from Keras.
- Rescaling is applied to normalize pixel values between 0 and 1.

## Model Architecture

The CNN model is defined using the following layers:

1. Convolutional layer with 16 filters, kernel size (3,3), and ReLU activation.
2. MaxPooling layer with a pool size of (2,2).
3. Convolutional layer with 32 filters, kernel size (3,3), and ReLU activation.
4. MaxPooling layer with a pool size of (2,2).
5. Convolutional layer with 64 filters, kernel size (3,3), and ReLU activation.
6. MaxPooling layer with a pool size of (2,2).
7. Flatten layer to transform the 3D output to 1D.
8. Dense layer with 512 neurons and ReLU activation.
9. Output layer with 1 neuron and Sigmoid activation for binary classification.

## Model Compilation

- Binary crossentropy loss is used for binary classification.
- RMSprop optimizer with a learning rate of 0.001 is employed.
- Accuracy is chosen as the evaluation metric.

## Model Training

The model is trained using the training dataset with early stopping to prevent overfitting. The training process involves 50 epochs, and the validation dataset is used to monitor model performance.

## Testing the Model

The trained model is tested on new images from the "basedata/testing" directory. Each image is loaded, resized to (200,200), and classified as either a cat or a dog.

## Usage

1. Ensure all dependencies are installed.
2. Organize your dataset with training, validation, and testing directories.
3. Adjust file paths in the provided script to match your dataset structure.
4. Run the script to train the model and test it on new images.
