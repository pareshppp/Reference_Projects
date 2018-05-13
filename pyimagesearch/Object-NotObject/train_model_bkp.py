# References:
# https://www.pyimagesearch.com

# set matplotlib backend so figure can be saved in the background
import matplotlib
matplotlib.use('Agg')

# import required packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

from cnn_models import LeNet



# construct argument parse and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, \
                help='path to input dataset')
ap.add_argument('-m', '--model', required=True, \
                help='path to output model')
ap.add_argument('-p', '--plot', type=str, default='plot.png', \
                help='path to output accuracy/loss plot')
args = vars(ap.parse_args())



# initialize variables
EPOCHS = 25    # number of epochs
INIT_LR = 1e-3  # initial learning rate
BS = 32 # image batch size

print('[INFO] loading images...')
data = []   # store image data
labels = [] # store image labels 

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args['dataset'])))
random.seed(123)
random.shuffle(imagePaths)



# loop over the images
for imagePath in imagePaths:
    # load the image, pre-process it and store it in data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))  # required size for LeNet
    image = img_to_array(image)
    data.append(image)

    # extract class labels from the image path and add to labels
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == 'Zoro' else 0
    labels.append(label)



# scale pixel intensities to range [0,1]
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)


# split train (75%) and test data (25%)
xtrain, xtest, ytrain, ytest = train_test_split(data, labels, \
                                    test_size=0.25, random_state=123)


# one-hot-encode labels
ytrain = to_categorical(ytrain, num_classes=2)
ytest = to_categorical(ytest, num_classes=2)


# construct image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, 
            height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode='nearest')


# initialize the model
print('[INFO] compiling model...')
model = LeNet.build_model(width=28, height=28, depth=3, nclasses=2)
optm = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=optm, metrics=['accuracy'])


# train the network
print('[INFO] training model...')
fit_gen = model.fit_generator(aug.flow(xtrain, ytrain, batch_size=BS),\
                    validation_data=(xtest, ytest), \
                    steps_per_epoch=len(xtrain)//BS,\
                    epochs=EPOCHS, verbose=1)


# save model to disk
print('[INFO] serializing model...')
model.save(args['model'])



# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
N = EPOCHS
H = fit_gen
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy on Zoro/notZoro')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(args['plot'])



