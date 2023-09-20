# Vertebras_Landmarking_using_Resnet

This project focuses on the task of marking keypoints on each vertebra, specifically the **IL (Left Inferior), IR (Right Inferior), SL (Left Superior), and SR (Right Superior)** points. The dataset comprises both scoliosis and normal vertebrae images. Each image is associated with a label file, with a total of 480 training images and 120 test images. The image files have a ".jpg" extension, while the label files use the ".mat" extension. Data augmentation techniques, such as resizing, random brightness/contrast adjustments, sharpening, adding multiplicative noise, image inversion, and vertical flipping, have been applied to augment the training data, increasing it to 2880 samples.

## Data Preprocessing
Before training the model, the data undergoes preprocessing, which involves the following steps:

  Conversion of RGB images to grayscale.
  
  Equalization of image intensities.
  
  Noise reduction.
  
  Normalization of images.
  
The preprocessed data is then serialized into a pickle file for efficient storage and retrieval.

## Hyperparameter Optimization
Bayesian optimization is applied to find optimal hyperparameters for training the model. The hyperparameters optimized include:

  Optimizer choice.
  
  Batch size.
  
  Number of epochs.
  
  Loss function.
  
## Model Architecture
The model used for this project is **ResNet-101**, a deep neural network architecture known for its excellent performance on various computer vision tasks.

## Overfitting Prevention
To prevent overfitting during training, early stopping is implemented, allowing the model to halt training when it no longer improves on the validation set.

## Deployment
The trained model is deployed using **Tkinter**, a Python GUI library. The deployed application allows users to upload images, process them, and make predictions regarding the keypoints on vertebrae.

## How to Run the Test File
To run the test file, execute the following command:

    python3 TestScoliosisModel.py

## Requirements
Before running the project, make sure to install the required Python packages:

    pip install tensorflow
    pip install tkinter
Please ensure that you have these packages installed to run the project successfully.

Feel free to explore the code and make improvements as needed. If you have any questions or encounter issues, don't hesitate to reach out. Happy coding!

