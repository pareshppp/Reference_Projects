# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pandas as pd
import argparse
from imutils import paths
import cv2


def construct_argument_parser():
    # construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True, \
                    help='path to the trained model')
    ap.add_argument('-d', '--dataset', required=True, \
                    help='path to input dataset -- image folder')
    ap.add_argument('-r', '--resize', type=int, default=28, \
                    help='image resize shape -- number of pixels in any one dimension')
    ap.add_argument('-o', '--object', type=str, default='Zoro', \
                    help='classification object')
    args = vars(ap.parse_args())
    return args



class ModelTesting:

    def __init__(self):
        pass


    def get_imagePaths(self, dataset_path):
        # grab the image paths
        imagePaths = sorted(list(paths.list_images(dataset_path)))
        return imagePaths


    def preprocess_image(self, imagePath, resize_shape=(28, 28)):
        # pre-process the image for classification
        image = cv2.imread(imagePath)
        image = cv2.resize(image, resize_shape) # resize
        image = image.astype('float') / 255.0   # normalize
        image = img_to_array(image)             # to array
        image = np.expand_dims(image, axis=0)   # add batch dim
        return image


    def classify_image(self, image, clf_object):
        # classify the input image
        (notObject, Object) = model.predict(image)[0]

        # get the prediction
        pred = 1 if Object > notObject else 0
        return pred






if __name__ == '__main__':

    predictions = []
    accuracy_df = pd.read_csv('true_labels.csv')

    # construct argument parser and parse arguments
    args = construct_argument_parser()

    # classification object
    clf_object = args['object'] 

    # create class object
    MT = ModelTesting()

    # load the trained cnn model
    print('[INFO] loading model...')
    model = load_model(args['model'])

    # grab the image paths
    imagePaths = MT.get_imagePaths(dataset_path=args['dataset'])


    # loop over all imagePaths and get prediction
    for imagePath in imagePaths:
        # load the image, pre-process it and store it in data list
        preprocessed_image = MT.preprocess_image(imagePath, \
                                    resize_shape=(args['resize'], args['resize']))
        
        # classify the input image and build the label
        pred = MT.classify_image(preprocessed_image, clf_object)
        predictions.append(pred)
    

    # compare predictions with true labels
    accuracy_df['predicted_labels'] = np.array(predictions)
    accuracy_df['match'] = accuracy_df['true_labels'] == accuracy_df['predicted_labels']

    accuracy = sum(accuracy_df['match']) / len(accuracy_df['match'])
    
    print('Testing Accuracy = {}'.format(accuracy))