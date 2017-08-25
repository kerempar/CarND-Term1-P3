import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

print("version 0.28")

def generator(images, measurements, batch_size=32):
    num_samples = len(images)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(images, measurements)
        for offset in range(0, num_samples, batch_size):
            batch_images = []
            batch_angles = []
            #print("offset:"+str(offset))
            for j in range(batch_size):
            	index = offset+j 
            	if index<num_samples:
                	image = images[index]
                	angle = measurements[index]
                	batch_images.append(image)
                	batch_angles.append(angle)

            X_train = np.array(batch_images)
            y_train = np.array(batch_angles)
            yield sklearn.utils.shuffle(X_train, y_train)
 
images = []
measurements = []

print("Reading lines from Udacity data: " + "./udacity_data/driving_log.csv")

lines = []

with open('./udacity_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)  # skip the headers
	for line in reader:
		lines.append(line)

print("Number of lines from Udacity data: " + str(len(lines)))

print("Reading images from: " + "./udacity_data/IMG")

j=0
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = './udacity_data/IMG/' + filename
		
		image = cv2.imread(current_path)
		RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		images.append(RGB_image)
		
		# save flipped image for documentation purposes
		#flipped_image = cv2.flip(image,1)
		#current_path = './udacity_data/IMG_FLIPPED/' + filename
		#cv2.imwrite(current_path,flipped_image)
		
		# get and adjust steering angle for right and left images
		measurement = float(line[3])
		if i==1: #	left image	
			measurement = measurement + 0.2
		if i==2: #	right image	
			measurement = measurement - 0.2
		
		measurements.append(measurement)
		j=j+1
		
print("Number of images added from Udacity data: " + str(j))

print("Reading lines from my data: " + "./my_data/driving_log.csv")

lines = []

with open('./my_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)  # skip the headers
	for line in reader:
		lines.append(line)

print("Number of lines from my data: " + str(len(lines)))

print("Reading images from my data: " + "./my_data/IMG")

j = 0
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = './my_data/IMG/' + filename
	
		image = cv2.imread(current_path)
		RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		images.append(RGB_image)
		
		# save flipped image for documentation purposes
		#flipped_image = cv2.flip(image,1)
		#current_path = './my_data/IMG_FLIPPED/' + filename
		#cv2.imwrite(current_path,flipped_image)
		
		# get and adjust steering angle for right and left images
		measurement = float(line[3])
		if i==1: #	left image	
			measurement = measurement + 0.2
		if i==2: #	right image	
			measurement = measurement - 0.2
	
		measurements.append(measurement)
		j=j+1

print("Number of images added from my data: " + str(j))

print("Augmenting data with flipped images... ")

# data augmentation by flipping images
# having more data to train the network
# the data will be more comprehensive
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)

print("Number of data points after augmentation: " + str(len(augmented_images)))

# build the Keras model 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout

model = Sequential()

# preprocessing steps
# normalizing
# adding a lambda layer, dividing each element by 255, to a range btw 0 and 1
# mean centering
# substracting 0.5 from each element,shift the element mean from 0.5 to 0
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# cropping top 70 and bottom 25 rows of the image
model.add(Cropping2D(cropping=((70,25),(0,0))))

# implement NVIDIA architecture
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(1))
# minimize the error between steering measurement predicted and ground truth steering measurement
model.compile(loss='mse', optimizer='adam')

# training 
# split off 20% of the data to be used for validation set
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5, verbose=1)
from sklearn.model_selection import train_test_split
train_images, validation_images, train_angles, validation_angles = train_test_split(augmented_images, augmented_measurements, test_size=0.2)

print("Number of images in training set: " + str(len(train_images)))
print("Number of angles in training set: " + str(len(train_angles)))
print("Number of images in validation set: " + str(len(validation_images)))
print("Number of angles in validation set: " + str(len(validation_angles)))

# create generators for training and validation sets 
train_generator = generator(train_images, train_angles, batch_size=32)
validation_generator = generator(validation_images, validation_angles, batch_size=32)

history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_images), 
			validation_data=validation_generator, 
            nb_val_samples=len(validation_images), 
            nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

print(model.summary())

# save the model
model.save('model.h5')

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()



