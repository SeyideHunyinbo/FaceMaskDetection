# FaceMaskDetection
This is an image classification problem, and the aim is to perform face mask detection using the Convolution Neural Network (CNN) workflow in TensorFlow and OpenCV


# Dataset
* The dataset contains `train` (with 1315 images) and `test folders` (with 194 images), and each folder contains 2 subfolders (i.e., with_mask and without_mask), each having equal number of images for both train and test sets. A `with_mask` folder consists of images of people with mask, and the images of people without masks is stored in a `without_mask` folder.

# Files
This repo contains 6 files namely:
* `face_mask_detection2.ipynb`            : Python file for model development
* `implement_mask_detection.py`           : Python file for face mask detection 
* `facemaskdetectormodel2.h5`             : CNN model
* `haarcascade_frontalface_default.xml`   : Frontal face detector
