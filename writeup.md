# **Behavioral Cloning** 

## Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img1.jpg
[image2]: ./img2.jpg
[image3]: ./img3.jpg
[image4]: ./img4.jpg
[image5]: ./img5.jpg
[image6]: ./img6.jpg

# Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
I added one line of code to drive.py to convert the images from RGB to BGR because the model is trained 
with images imported with cv2.imread which reads images in BGR while drive.py is using RGB mode.

### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model is inspired from NVIDIA convulution neural network, it consists of a normalization layer and 5 convolutional layers followed by 4  fully connected layers (model.py lines 74-91) 

The model includes RELU layers to introduce nonlinearity (code line 80-84), and the data is normalized in the model using a Keras lambda layer (code line 76). 

### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 84). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 96-98). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 97).

## 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road data.

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try different architectures and to test and modify them.

My first step was to use a convolution neural network model similar to the Lenet I thought this model might be appropriate because it is powerfull for learning features in images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

After that I modified the model architecture and used a model simular to NVIDIA convulution neural network, to combat the overfitting I added a Dropout layer to it.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and it was not steering well to the right side. to improve the driving behavior in these cases, I augmented the training data by flipping the images and I also added images of recovering from the left and right sides.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture

The final model architecture (model.py lines 74-93) consisted of a convolution neural network with the following layers and layer sizes.
Layer 1: Normalization layer with input shape of (160, 320, 3)
Layer 2: Cropping layer to remove the irrelevant image features
Layer 3: Convulution RELU layer with size (24,5,5)
Layer 4: Convulution RELU layer with size (36,5,5)
Layer 5: Convulution RELU layer with size (48,5,5)
Layer 6: Convulution RELU layer with size (64,5,5)
Layer 7: 50% Dropeout layer
Layer 8: Convulution RELU layer with size (64,5,5)
Layer 9: Flatten Layer
Layer 10: fully connected Layer with 100 neurons
Layer 11: fully connected Layer with 50 neurons
Layer 12: fully connected Layer with 10 neurons

### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from left and right sides once it finds itself in such a case These images show what a recovery looks like:

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would let the model be able to learn steering well in both sides.

![alt text][image5]
![alt text][image6]



After the collection process, I had  24108 number of data points. I then preprocessed this data by cropping the irrelevant image features.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the training and validation loss values that stopped making important decrease after the forth epoch. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.
