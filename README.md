# CS6910 Assignment02
## Image Classification with CNNs on iNaturalist Dataset:

This project focuses on utilizing Convolutional Neural Networks (CNNs) to classify images in a subset of the iNaturalist dataset, which comprises a large collection of images depicting various plant and animal species.

**Part A: Building a 5-Layered CNN from Scratch**

In the first part of the project, a 5-layered CNN is constructed from scratch to classify images in the iNaturalist dataset. The trained model is then evaluated on a test set to measure its accuracy in correctly classifying the images.

**Part B: Transfer Learning using a Pre-Trained Model**

In the second part of the project, a pre-trained model (ResNet50) is employed for the same classification task. The aim of this part of the project is to demonstrate the concept of transfer learning, wherein a pre-trained model is fine-tuned to perform a specific task. The ResNet50 model is fine-tuned on the provided subset of the iNaturalist dataset, and its performance is evaluated on the test set.

The results show that the fine-tuned model is able to provide double the accuracy of the model trained from scratch, demonstrating the power of transfer learning.

Tools Used:
wandb 
torch 
torchvision
lightning module

CS6910_Assignment_02_Part_A.ipynb - Contains code for Part A of the assignment includes training and testing
CS6910_Assignment_02_Part_A_Testing.ipynb - Contains code for Part A of the assignment testing after finding the best hyperparameter

CS6910_Assignment_02_Part_B.ipynb - Contains code for Part B of the assignment

Once the preliminary version of the code has been developed in the Colab notebooks, it is transferred to the following Python files for training purposes

train_parta.py

train_partb.py

Evaluation Metrics:

The performance of the models is evaluated using accuracy as the metric. The accuracy is calculated as the ratio of correctly classified images to the total number of images in the test set.

Conclusion:

This project demonstrates the effectiveness of CNNs for image classification tasks and the power of transfer learning using pre-trained models. The results show that fine-tuning a pre-trained model can significantly improve the prediction accuracy in classifying images
