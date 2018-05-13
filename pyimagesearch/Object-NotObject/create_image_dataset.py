# References:
# https://github.com/hardikvasa/google-images-download
# https://www.pyimagesearch.com/

# !pip install google_images_download

import os
import cv2
from google_images_download import google_images_download as gid


def construct_argument_parser():
    # construct argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', '--keywords', required=True, \
                    help='keyword to search for')
    ap.add_argument('-l', '--limit', type=int, required=True,
                    help='number of search results')
    args = vars(ap.parse_args())
    return args

class create_image_dataset:

    # initialize constructor
    def __init__(self, keyword, limit):
        self.keyword = keyword
        self.limit = limit
        self.directory = os.path.join(os.path.dirname(__file__), "images")
        self.chromedriver = '/usr/local/share/chromedriver'

    # download images from google image search
    def get_images(self):
        # class instantiation
        response = gid.googleimagesdownload()

        # creating dictionary of script arguments
        arguments = {
            'keywords': self.keyword,  # image search keyword
            'limit': self.limit,  # number of search results
            'output_directory': self.directory,
            'print_urls': True,
            'format': 'jpg',
            'chromedriver': self.chromedriver
        }  # creating list of arguments

        # passing the arguments to the function
        response.download(arguments)

    # remove unreadable or corrupt images by reading them using cv2.imread()
    def clean_images(self):
        # image_dir = script_path/images/"keyword"
        image_dir = os.path.join(self.directory, self.keyword)

        print("Reading Images with OpenCV:\n")

        # loop over images in directory
        for file in os.listdir(image_dir):
            imagepath = os.path.join(image_dir, file)

            # set delete flag to False in every iteration
            delete = False

            # read image:
            # if successful -> keep
            # if error -> delete
            try:
                print(imagepath)

                # read image in cv2 as single channel
                image = cv2.imread(imagepath, 0)

                # if image is None, then it could not be read properly
                # so delete it
                if image is None:
                    print('Empty File.')
                    delete = True

            except:
                # file is probably corrupt, so delete it
                print('Error Reading File.')
                delete = True

            # delete image if delete is true
            if delete:
                print("[INFO] deleting {}".format(imagepath))
                os.remove(imagepath)

    def rename_images(self):
        image_dir = os.path.join(self.directory, self.keyword)
        i = 1

        for file in os.listdir(image_dir):  # loop over images in directory
            imagepath = os.path.join(image_dir, file)
            os.rename(imagepath,
                      os.path.join(image_dir, 'img' + str(i) + '.jpg'))
            i += 1
        print('Renaming Images - Complete')


if __name__ == "__main__":
    import argparse

    # adding command line arguments
    args = construct_argument_parser()

    # executing all functions
    print("Creating Image Dataset:- Start")
    dataset = create_image_dataset(args['keywords'], args['limit'])
    dataset.get_images()
    dataset.rename_images()
    dataset.clean_images()
    print("Creating Image Dataset:- Complete")