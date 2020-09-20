"""
Vehicle Type Recognition. Classification
of images of different vehicle types, including cars,
bicycles, vans, ambulances, etc. (total 17 categories).
Prediction script contains most of the initialization and prediction.
"""

__author__ = "Niranjan Hegde"
__copyright__ = "Copyright 2020"
__license__ = "All rights reserved"
__maintainer__ = "Niranjan Hegde"
__email__ = "niranjan.hegde@gmail.com"
__status__ = "Beta test"


import argparse
from vehicle.postprocessing import generate_submit
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():
    # Parameters
    parser = argparse.ArgumentParser(description='Vehicle Recognition Competition PREDICT')
    parser.add_argument('--model', default='IncResNet-ep69-0.03-0.99', type=str, help='Model name')
    parser.add_argument('--test', default='./test/', type=str, help='Directory of test data')
    parser.add_argument('--IMG_HEIGHT', default='224', type=int, help='Image height')
    parser.add_argument('--IMG_WIDTH', default='224', type=int, help='Image width')
    parser.add_argument('--batch_size', default='1', type=int, help='Batch size')

    args = parser.parse_args()

    # Generator for the test data
    test_generator = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_generator.flow_from_directory(args.test, target_size=(args.IMG_HEIGHT, args.IMG_WIDTH),
                                                        shuffle=False, batch_size=args.batch_size)

    # Predict values
    generate_submit(args, test_generator)


if __name__ == "__main__":
    main()

