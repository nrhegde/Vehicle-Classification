import glob
import os
import numpy as np

from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class VehicleDataset():
    def __init__(self, args):
        # Data Generator for the train data
        data_generator = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, horizontal_flip=True, rotation_range=10,
                                            width_shift_range=.1, height_shift_range=.1)

        self.train_generator = data_generator.flow_from_directory(args.train, target_size=(args.IMG_HEIGHT, args.IMG_WIDTH),
                                                             shuffle=True,
                                                             seed=13,
                                                             class_mode='categorical', batch_size=args.batch_size,
                                                             subset="training")

        self.validation_generator = data_generator.flow_from_directory(args.train, target_size=(args.IMG_HEIGHT, args.IMG_WIDTH),
                                                                  shuffle=True, seed=13,
                                                                  class_mode='categorical', batch_size=args.batch_size,
                                                                  subset="validation")

    def load_data(self, folder):
        """
        Load all images from subdirectories of
        'folder'. The subdirectory name indicates
        the class.
        """

        X = []          # Images go here
        y = []          # Class labels go here
        classes = []    # All class names go here

        subdirectories = glob.glob(folder + "/*")

        # Loop over all folders
        for d in subdirectories:

            # Find all files from this folder
            files = glob.glob(d + os.sep + "*.jpg")

            # Load all files
            for name in files:

                # Load image and parse class name
                img = imread(name)
                class_name = name.split(os.sep)[-2]

                # Convert class names to integer indices:
                if class_name not in classes:
                    classes.append(class_name)

                class_idx = classes.index(class_name)

                X.append(img)
                y.append(class_idx)

        # Convert python lists to contiguous numpy arrays
        X = np.array(X)
        y = np.array(y)
        classes = np.array(classes)

        return X, y, classes

    def resize_vehicle_test_data(self):
        root = './test/testset/'
        new_path = './test/scaled_test/'
        os.mkdir(new_path)
        i = 0
        for file in sorted(os.listdir(root)):
            img = imread(root + file)
            res = resize(img, (200, 200))
            imsave(new_path + os.sep + file, img_as_float(res))
            i = i + 1
            print(str(i) + ' images out of ' + str(len(os.listdir(root))) + ' processed')

        print('Successfully resized')

    def resize_vehicle_train_data(self):
        root = './train/train/'
        new_path = './train/scaled_train/'
        os.mkdir(new_path)
        i = 0
        # Rescale all files in each subdirectory
        for category in sorted(os.listdir(root)):
            os.mkdir(new_path + category)
            for file in sorted(os.listdir(os.path.join(root, category))):
                img = imread(root + category + os.sep + file)
                res = resize(img, (200, 200))
                imsave(new_path + os.sep + category + os.sep + file, img_as_float(res))
                i = i + 1
                print(str(i) + ' images processed')

        print('Successfully resized')
