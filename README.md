# Face Image Classificaiton using Convolution Neural Networks (CNN)

DataSet used: Labeled Faces in the Wild (LFW) dataset

In this project, we compare the performance of the CNN architecture for face image classification using different hyperparameters. It investigates the relationship between the various hyperparameters [learning rate and dropout regularization rate] and the performance of the learned model. In a series of experiments we also determine the performance of the trained model with varying epoch size.

For training, m=13200 images were used and for testing n=1000 images were used. All images were cropped to 60*60 pixels. For non-faces, top left corner of 60*60 resolution was cropped. The images were converted into grayscale.
