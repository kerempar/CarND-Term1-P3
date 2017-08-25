#**Behavioral Cloning** 

##Kerem Par

###kerempar@gmail.com

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/nvidia-cnn-architecture.png =300x450 "Model Visualization"
[image2]: ./writeup_images/center_2016_12_01_13_30_48_287.jpg =500x200 "Center line driving"
[image3]: ./writeup_images/left_2016_12_01_13_30_48_287.jpg =250x100 "Left Image"
[image4]: ./writeup_images/right_2016_12_01_13_30_48_287.jpg =250x100 "Right Image"
[image5]: ./writeup_images/center_2016_12_01_13_39_38_978.jpg =250x100 "Normal Image"
[image6]: ./writeup_images/center_2016_12_01_13_39_38_978_flipped.jpg =250x100 "Flipped Image"
[image7]: ./writeup_images/model_summary.png =500x300 "Network architecture"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter and 5x5 filter sizes and depths between 24 and 64 (model.py lines 150-165) 

The model includes RELU layers to introduce nonlinearity (code line 152-156), and the data is normalized in the model using a Keras lambda layer (code line 146). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 159, 161, 163). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 180-183). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 166).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to build the most basic network possible first, then improve the network iteratively and observe the results.
The initial network was a flattened image connected to a single output node - to predict the steering angle. 

Then I implemented the LeNet architecture and finally I implemented NVIDIA's architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

To combat the any overfitting, I modified the model by adding three dropout layers. 

The testing step was to run the simulator to see how well the car was driving around track one. I found that my first and second models did not succeed completely. There were a some spots where the vehicle fell off the track. I observed a significant improvement with the NVIDIA's architecture. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 150-165) consisted of a convolution neural network that NVIDIA uses to drive its autonmous car with addition of some dropout layers.

Here is a visualization of the NVIDIA's original architecture. 

![alt text][image1]

Here is a visualization of my final architecture displayed by using Keras's model.summary function.

![alt text][image7]

####3. Creation of the Training Set & Training Process

I first used the Udacity's data which is recorded on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

The data set contains 8036 lines. I also included left and right images, so I had a total of 24108 data points. 

![alt text][image3]
![alt text][image4]

I then recorded one lap on track one using the simulator including both center line driving and also recovering from the left side and right sides of the road back to center. My data contains 
3560 lines. Similarly, I included left and right images, so I had a total of 10680 data points. 

I combined both data sets and had a data set of 34788 data points. 

To augment the data set, I also flipped images and angles. So, I ended up a final data set of 69576 data points. For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]

I finally randomly shuffled the data set and put 20% of the data into a validation set (Training set: 55660, Validation Set: 13915 data points).

I then preprocessed this data by normalizing (dividing each element by 255, to a range between 0 and 1) and mean centering (substracting 0.5 from each element, shift the element mean from 0.5 to 0). I also applied cropping of top 70 rows and bottom 25 rows of each image.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
