# CnnAutoencoder
A convolutional autoencoder, used to compress and deblur various images of cars found on https://www.kaggle.com/datasets/kshitij192/cars-image-dataset
With this structure, a compression rate of 92.68% can be achieved, while the reconstructed image is 98.33% similar to the original image.
For deblurring puproses, a GaussianBlur transformer is used, and the cleared image is on average 96.1% similar to the clear image.
The images were first scaled to 256x256 rgb, before being fed to the network. A simple fuly connected feed forward network would require 196.608 input neurons (3*256*256, 3 because the images are rgb).
