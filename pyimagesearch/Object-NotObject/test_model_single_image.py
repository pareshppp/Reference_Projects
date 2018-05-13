# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2


def construct_argument_parser():
    # construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True, \
                    help='path to the trained model')
    ap.add_argument('-i', '--image', required=True, \
                    help='path to input image')
    ap.add_argument('-r', '--resize', type=int, default=28, \
                    help='image resize shape -- number of pixels in any one dimension')
    ap.add_argument('-o', '--object', type=str, default='Zoro', \
                    help='classification object')
    args = vars(ap.parse_args())
    return args



class ModelTesting:

    def __init__(self):
        pass

    def preprocess_image(self, image, resize_shape=(28, 28)):
        # pre-process the image for classification
        image = cv2.resize(image, resize_shape) # resize
        image = image.astype('float') / 255.0   # normalize
        image = img_to_array(image)             # to array
        image = np.expand_dims(image, axis=0)   # add batch dim
        return image


    def classify_image(self, image, clf_object):
        # classify the input image
        (notObject, Object) = model.predict(image)[0]

        # build the label
        label = clf_object if Object > notObject else 'Not ' + clf_object
        proba = Object if Object > notObject else notObject
        label = '{}: {:.2f}%'.format(label, proba * 100)
        return label


    def draw_label_on_image(self, orig, label):
        # draw the label on the image
        output = imutils.resize(orig, width=400)
        cv2.putText(output, label, (10, 25), \
                    cv2.FONT_HERSHEY_SIMPLEX,\
                    0.7, (0, 255, 0), 2)
        return output









if __name__ == '__main__':

    # construct argument parser and parse arguments
    args = construct_argument_parser()

    # classification object
    clf_object = args['object'] 

    # create class object
    MT = ModelTesting()

    # load the image and create a backup of original
    image = cv2.imread(args['image'])
    orig = image.copy()

    # pre-process the image for classification
    image = MT.preprocess_image(image, resize_shape=(args['resize'], args['resize']))

    # load the trained cnn model
    print('[INFO] loading model...')
    model = load_model(args['model'])

    # classify the input image and build the label
    label = MT.classify_image(image, clf_object)

    # draw the label on the image
    output = MT.draw_label_on_image(orig, label)

    # show the output image
    cv2.imshow('Output', output)
    cv2.waitKey(0)
