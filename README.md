# Deep-Learning-projects
A collection of projects in the Deep learning and Machine Learning domain

# Index

1. Backpropagation Algorithm
2. Implementation of Decision Tree and Naive Bayes algos for multi-class classification
3. Face Image classification using Convolution Neural Networks

______________________________________________________________________________________________________________________________
1. Forward and Backpropagation algorithm for a Multi Layer Perceptron (MLP)

DataSet used: MNIST dataset

For performing image classification on the multi-class MNIST dataset, we implemented the multilayer perceptron using the sigmoid activation function. We tested several different network architectures before selecting our final architecture for this task. Later we tuned the network to obtain the optimal hyperparamters. 

This network was then used to analyse the accuracy and loss results on the training, validation, and testing datasets.

Results obtained: 
For Training Datatset: Accuracy=90.866
For Validation Dataset: Accuracy=89.49
Achieved an 88.48% testing accuracy using MLP. This in comparison to CNNs is abysmal where we achieve over 99% accuracy. 


______________________________________________________________________________________________________________________________
2. Implementation of Decision Tree (DT) and Naive Bayes (NB) Algorithms to classify Multi-class Images 

DataSet used: Yale B and MNIST dataset

Implemented the DT and NB algorithms using numpy as the only dependency and performed 5-fold cross-validation to choose the best hyper parameteres. The two algorithms were then tested on both the datasets.

Bagging was also performed and we observed that the model's performance improved with bagging in comparison to a classical DT.
We also implemented pre-pruning on the decision tree algorithm.



______________________________________________________________________________________________________________________________
3. Face Image Classificaiton using Convolution Neural Networks (CNN)

DataSet used: Labeled Faces in the Wild (LFW) dataset

In this project, we compare the performance of the CNN architecture for face image classification using different hyperparameters. It investigates the relationship between the various hyperparameters [learning rate and dropout regularization rate] and the performance of the learned model. In a series of experiments we also determine the performance of the trained model with varying epoch size.

For training, m=13200 images were used and for testing n=100 images were used. All images were cropped to 60*60 pixels. For non-faces, top left corner of 60*60 resolution was cropped. The images were converted into grayscale.
