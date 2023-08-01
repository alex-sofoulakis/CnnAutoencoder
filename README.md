# CnnAutoencoder
A convolutional autoencoder, used to compress and deblur various images of cars ( found on https://www.kaggle.com/datasets/kshitij192/cars-image-dataset
With this structure, a compression rate of 92.68% can be achieved, while the reconstructed image is 98.33% similar to the original image.
For deblurring puproses, a GaussianBlur transformer is used, and the cleared image is on average 96.1% similar to the clear image.
The images were first scaled to 256 by 256 rgb, before being fed to the network. A simple fuly connected feed forward network would require 196.608 input neurons (3 by 256 by 256, 3 because the images are rgb).
The sliding kernel logic of a CNN solves this problem, as it requires far less computational resources.
The batch_size and epochs were set to 14 and 35 respectively, after some minimal testing. To determine the learning rate, more thorough tests were conducted, settling on lr = 1e-3. 
The loss function used was MSE, suitable for comparing the difference in image pixels. The best optimizer was Adam, providing a faster convergance. 
The code can be found both as a .py file and a ipynb file (the data needs to be on google drive first).

Example of dublurring results

![image](https://github.com/thatweirdboi93/CnnAutoencoder/assets/56234672/9028f130-c7fa-4edd-828c-4c5e7ef2d0ce)

Example of simple compression and decompression

![image](https://github.com/thatweirdboi93/CnnAutoencoder/assets/56234672/e5cc1545-c9d0-474d-aaed-049ced3cc5c0)
