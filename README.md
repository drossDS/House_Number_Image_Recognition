# House Number Image Recognition
Employing Neural Networks to Recognize Digits

## Background and Problem Statement
The Street View Housing Numbers (SVHN) dataset contains over 600,000 images of house numbers captured from Google Street View Images.  Each image is 32 x 32 pixels and is identified as the correct digit.  The data provided for this project were gray-scale images dervied from the SVHN dataset, pared back to only 60,000 images in total (presumaly to make this much less computationally intensive).  The goal of this project was to create a model which could identify the images from this dataset as their correct digit.  To do this, both artificial and convolutional neural networks (ANN and CNN) algorithms were employed.

## Initial Processing and Exploratory Data Analysis


*Note:  Due to the computationally intensive nature of this model and the large amount of data, this project was completed in Google Colab so that the built-in GPU could be utilized to decrease processing time. It should also be noted that the ANN and CNN models were run in separate notebooks.  Given that the steps taken for each model were very similar, this project overview will describe the models as if they were all completed in a single notebook, and describe the differences where appropriate.*

***DAN - GO BACK AND DOWNLOAD A NEW VERSION OF THE CODE AS A JUPYTER NOTEBOOK AND DON'T DO THE HTML THING

The provided dataset was pre-split into training and test datasets of 42,000 and 18,000 images respectively.  These were each assigned to their own dataframes.  Examining the images revealed hat each image was stored as a 32 x 32 matrix with values between 0 and 255 (indicating color/darkness).  For the ANN model, the 32 x 32 matrices representing each image were reshaped to a singe array of 1024 elements.  This was not done for the data being processed by the CNN model, and presumably, this is to maintain the 2-dimensional proeprties of the image so that the convolution fileters could be applied.  The data being used in both models were then normalized by dividing all elements by 255 so that all numerical values would range proprotionally between 0 and 1 for neural network algorithms.  The image label datasets (ranging between between 0 and 9) were then one-hot encoded so that each digit could be mapped to a category into which the nerual networks could assign the provided image data.

## Models Run

All Artificial and Convolutional Neural Network models were run with Keras as sequential models.  For both ANN and CNN models, two variations were run to showcase the benefits of adding features to improve model accuracy.  By nature, the ANN models were far simpler than the CNN models, and this is reflected specifically in the both quantities and types of layers that make up each model.  Below are brief descriptions of each along with tables of the detailed properties of each model.  The bolded values in tables below highlight the differences between the properties of the two model variations for the ANN and CNN models.

### Artificial Neural Network (ANN) Model Properties
ANN Model 1 implmented two hidden layers employing ReLU activation funcitons.  Model 2 added more hidden layers, dropout layers, and batch normalization layers, and increased the number of nodes speified in each hidden layer.  The models are compared side-by-side in the table below for easy comparison.

<details>
  
  <summary>
    <b><i>TABLE - Artificial Neural Network Properties (Click for Dropdown)</i></b>
    </summary>
  
  <br>
  
| | Layer | **Model 1 Properties** || | Layer | **Model 2 Properties** |
| --- | --- | ---|---| --- | --- | ---|
|1| Hidden | Nodes: 64, Activation: Relu ||1| Hidden | ***Nodes: 256***, Activation: Relu |
|2| Hidden | Nodes: 32, Activation: Relu ||2| Hidden | ***Nodes: 128***, Activation: Relu |
|||||***3***| ***Dropout*** | ***Dropout Rate: 0.2*** |
|||||***4***| ***Hidden*** | ***Nodes: 64, Activation: Relu*** |
|||||***5***| ***Hidden*** | ***Nodes: 64, Activation: Relu*** |
|||||***6***| ***Hidden*** | ***Nodes: 32, Activation: Relu*** |
|||||***7***| ***Batch Normalization*** | ***None Specified*** |
|3| Output | Nodes: 10, Activiation: Softmax ||8| Output | Nodes: 10, Activiation: Softmax |
| <br>|  |  || |  |  |
|-| Compile | Loss: Categorical Crossentropy<br>Optimizer: Adam, Learning Rate: 0.001<br>Metrics: Accuracy ||- | Compile | Loss: Categorical Crossentropy<br>Optimizer: Adam, ***Learning Rate: 0.0005***<br>Metrics: Accuracy |
|-| Fitting | Validation Split: 0.2<br>Batch Size: 128<br>Epochs: 20 ||- | Fitting | Validation Split: 0.2<br>Batch Size: 128<br>***Epochs: 30*** |
  </details>

### Convolutional Neural Network (CNN) Model Properties
CNN model 1 employed convolutional filters with 3x3 kernels and LeakyReLU activation layers in addition to max-pooling, flattening, and dense layers.  Model 2 added more convolutional filter layers, along with dropout, and batch normalization layers.  The models are compared side-by-side in the table below for easy comparison.

<details>
  
  <summary>
    <b><i>TABLE - Convolutional Neural Network Properties (Click for Dropdown)</i></b>
    </summary>
  
  <br>

| | Layer | **Model 1 Properties** || | Layer | **Model 2 Properties** |
| --- | --- | ---|---| --- | --- | ---|
|1| Convolutional | Filters: 16, Kernel Size: 3x3, Padding: Same ||1| Convolutional | Filters: 16, Kernel Size: 3x3, Padding: Same |
|2| LeakyRelu | Slope: 0.1 ||2| LeakyRelu | Slope: 0.1 |
|3| Convolutional | Filters: 32, Kernel Size: 3x3, Padding: Same ||3| Convolutional | Filters: 32, Kernel Size: 3x3, Padding: Same |
|4| LeakyRelu | Slope: 0.1 ||4| LeakyRelu | Slope: 0.1 |
|5| Max Pooling | Pool Size:  2x2 ||5| Max Pooling | Pool Size:  2x2 | 
|||||***6***| ***Batch Normalization*** | ***None Specified*** |
|||||***7***| ***Convolutional*** | ***Filters: 32, Kernel Size: 3x3, Padding: Same*** |
|||||***8***| ***LeakyRelu*** | ***Slope: 0.1*** |
|||||***9***| ***Convolutional*** | ***Filters: 64, Kernel Size: 3x3, Padding: Same*** |
|||||***10***| ***LeakyRelu*** | ***Slope: 0.1*** |
|||||***11***| ***Max Pooling*** | ***Pool Size:  2x2*** |
|||||***12***| ***Batch Normalization*** | ***None Specified*** |
|6| Flatten | None Specified ||13| Flatten | None Specified |
|7| Dense | Nodes: 32 ||14| Dense | Nodes: 32 |
|8| LeakyRelu | Slope: 0.1 ||15| LeakyRelu | Slope: 0.1 |
|||||***16***|***Dropout*** | ***Dropout Rate: 0.5***|
|9| Output | Nodes: 10, Activiation: Softmax ||17| Output | Nodes: 10, Activiation: Softmax |
| <br>|  |  || |  |  |
|-| Compile | Loss: Categorical Crossentropy<br>Optimizer: Adam, Learning Rate: 0.001<br>Metrics: Accuracy ||- | Compile | Loss: Categorical Crossentropy<br>Optimizer: Adam, Learning Rate: 0.001<br>Metrics: Accuracy |
|-| Fitting | Validation Split: 0.2<br>Batch Size: 32<br>Epochs: 20 ||- | Fitting | Validation Split: 0.2<br>***Batch Size: 128***<br>***Epochs: 30*** |
</details>


## Conclusions


