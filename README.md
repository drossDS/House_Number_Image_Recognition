# House Number Image Recognition
Employing Neural Networks to Recognize Digits

## Background and Problem Statement
The Street View Housing Numbers (SVHN) dataset contains over 600,000 images of house numbers captured from Google Street View Images.  Each image is 32 x 32 pixels and is identified as the correct digit.  The data provided for this project were gray-scale images dervied from the SVHN dataset, pared back to only 60,000 images in total (presumaly to make this much less computationally intensive).  The goal of this project was to create a model which could identify the images from this dataset as their correct digit.  To do this, both artificial and convolutional neural networks (ANN and CNN) algorithms were employed.

## Initial Processing and Exploratory Data Analysis


*Note:  Due to the computationally intensive nature of this model and the large amount of data, this project was completed in Google Colab so that the built-in GPU could be utilized to decrease processing time. It should also be noted that the ANN and CNN models were run in separate notebooks.*

***DAN - GO BACK AND DOWNLOAD A NEW VERSION OF THE CODE AS A JUPYTER NOTEBOOK AND DON'T DO THE HTML THING

The provided dataset was pre-split into training and test datasets of 42,000 and 18,000 images respectively.  These were each assigned to their own dataframes.  Examining the images revealed hat each image was stored as a 32 x 32 matrix with values between 0 and 255 (indicating color/darkness).  The data were then normalized by dividing by 255 so that all numerical values would range proprotionally between 0 and 1 so that it could be processed by the neural network models.

## Models Run


### Convolutional Neural Network (CNN) Model Properties

| |  | **Model 1** || |  | **Model 2** |
| --- | --- | ---|---| --- | --- | ---|
| | Layer | Properties || | Layer | Properties |
|1| Convolutional | Filters: 16, Kernel Size: 3x3, Padding: Same ||1| Convolutional | Filters: 16, Kernel Size: 3x3, Padding: Same |
|2| LeakyRelu | Slope: 0.1 ||2| LeakyRelu | Slope: 0.1 |
|3| Convolutional | Filters: 32, Kernel Size: 3x3, Padding: Same ||3| Convolutional | Filters: 32, Kernel Size: 3x3, Padding: Same |
|4| LeakyRelu | Slope: 0.1 ||4| LeakyRelu | Slope: 0.1 |
|5| Max Pooling | Pool Size:  2x2 ||5| Max Pooling | Pool Size:  2x2 | 
|||||6| Batch Normalization | None SPecified |
|||||7| Convolutional | Filters: 32, Kernel Size: 3x3, Padding: Same |
|||||8| LeakyRelu | Slope: 0.1 |
|||||9| Convolutional | Filters: 64, Kernel Size: 3x3, Padding: Same |
|||||10| LeakyRelu | Slope: 0.1 |
|||||11| Max Pooling | Pool Size:  2x2 |
|||||12| Batch Normalization | None SPecified |
|6| Flatten | None Specified ||13| Flatten | None Specified |
|7| Dense | Nodes: 32 ||14| Dense | Nodes: 32 |
|8| LeakyRelu | Slope: 0.1 ||15| LeakyRelu | Slope: 0.1 |
|||||16|Dropout | Dropout Rate: 0.5|
|9| Output | Nodes: 10, Activiation: Softmax ||17| Output | Nodes: 10, Activiation: Softmax |
| |  | || |  | |
| |  | **Model 1 Compiling** || |  | **Model 2 Compiling** |
| | Validation Split | 0.2 || | Validation Split | 0.2 |



DAN - Say something about all models being "sequential"

## Conclusions


