from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pickle

# this file abstracts away the preparing and loading of data sets

# loads the prima head pose files for the pitch measurement
#   @param data_path = string containing where to load data set from
#   @param prima_test_person_out = int value that states which person to use for testing
def load_prima_head_pose_pitch(data_path, prima_test_person_out):

    image_size = 64 # images are 32x32x3
    num_channels = 3 # RGB
    num_classes = 1 # regression problem

    pickle_directory = data_path + "prima_pitch_p" + str(prima_test_person_out) + "_out.pickle"
    
    with open(pickle_directory, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
        training_data = data[b'training_dataset']
        training_labels = data[b'training_label']
        testing_data = data[b'validation_dataset']
        testing_labels = data[b'validation_label']
        # signal to be garbage collected
        del data

    # reshape data
    training_data = training_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    training_labels = training_labels.reshape((-1, 1)).astype(np.float32)

    testing_data = testing_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    testing_labels = testing_labels.reshape((-1, 1)).astype(np.float32)

    return training_data, training_labels, testing_data, testing_labels, image_size, num_channels, num_classes

# loads the prima head pose files for the yaw measurement
#   @param data_path = string containing where to load data set from
#   @param prima_test_person_out = int value that states which person to use for testing
def load_prima_head_pose_yaw(data_path, prima_test_person_out):

    image_size = 64 # images are 32x32x3
    num_channels = 3 # RGB
    num_classes = 1 # regression problem

    pickle_directory = data_path + "prima_yaw_p" + str(prima_test_person_out) + "_out.pickle"
    
    with open(pickle_directory, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
        training_data = data[b'training_dataset']
        training_labels = data[b'training_label']
        testing_data = data[b'validation_dataset']
        testing_labels = data[b'validation_label']
        # signal to be garbage collected
        del data

    # reshape data
    training_data = training_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    training_labels = training_labels.reshape((-1, 1)).astype(np.float32)

    testing_data = testing_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    testing_labels = testing_labels.reshape((-1, 1)).astype(np.float32)

    return training_data, training_labels, testing_data, testing_labels, image_size, num_channels, num_classes


# function to load data set and parameters for CIFAR10
#   @param data_path = string containing where to load data set from
#   @param validation = bool that states whether to prepare a validation set
#   @param test_split = int value that states percentage to use for testing set
def load_CIFAR_10(data_path, validation, test_split):
    num_channels = 3 # RGB
    image_size = 32 # 32x32 images
    num_classes = 10 # 10 possible classes. info @ https://www.cs.toronto.edu/~kriz/cifar.html

    pickle_directory = data_path
    num_training_files = 5 # CIFAR10 training data is split into 5 files
    num_images_per_file = 10000 # there are 10000 images and labels per
    num_training_images_total = num_training_files * num_images_per_file
    num_testing_images_total = 10000

    # training images
    training_set = np.zeros(shape=[num_training_images_total, image_size, image_size, num_channels], dtype=float)
    # training class numbers as integers
    training_classes = np.zeros(shape=[num_training_images_total], dtype=int)
    # training class labels in one hot encoding
    training_labels = np.zeros(shape=[num_training_images_total, num_classes], dtype=int)

    # to reference where to store current batch of data in large array
    begin = 0

    # cycle each batch of training data
    for i in range(num_training_files):
        pickle_file = pickle_directory + "data_batch_" + str(i + 1)
            
        # load current batch
        with open(pickle_file, mode='rb') as file:
            data = pickle.load(file, encoding='bytes')
            images_batch = data[b'data']
            classes_batch = np.array(data[b'labels'])

            # reshape data
            images_batch = images_batch.reshape([-1, num_channels, image_size, image_size])
            images_batch = images_batch.transpose([0, 2, 3, 1])

            num_images = len(images_batch)
            end = begin + num_images

            # store current batch in required segment of entire array
            training_set[begin:end, :] = images_batch
            training_classes[begin:end] = classes_batch

            begin = end

    # convert training labels from integer format to one hot encoding
    training_labels = np.eye(num_classes, dtype=int)[training_classes]
            
    # testing class labels in one hot encoding
    testing_labels = np.zeros(shape=[num_testing_images_total, num_classes], dtype=int)

    # load testing data
    pickle_file = pickle_directory + "test_batch"
    with open(pickle_file, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
        testing_set = data[b'data']
        testing_classes = np.array(data[b'labels'])

        # reshape data
        testing_set = testing_set.reshape([-1, num_channels, image_size, image_size])
        testing_set = testing_set.transpose([0, 2, 3, 1])

        # mark to be garbage collected
        del data

    # convert testing set labels from integer format to one hot encoding
    testing_labels = np.eye(num_classes, dtype=int)[testing_classes]

    # reshape data 
    training_set = training_set.reshape(-1, image_size, image_size, num_channels).astype(np.float32)
    testing_set = testing_set.reshape(-1, image_size, image_size, num_channels).astype(np.float32)

    # if validation set is required, partition segment of test data
    if validation == True:
        testing_set, testing_labels, validation_set, validation_labels = create_validation_set(testing_set, testing_labels, test_split)
    # otherwise just create an empty array
    else: validation_set, validation_labels = np.empty(shape=(2, 2))
        
    return training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels, image_size, num_channels, num_classes


# function to load data set and parameters for MNIST
#   @param data_path = string containing where to load data set from
#   @param validation = bool that states whether to prepare a validation set
#   @param test_split = int value that states percentage to use for testing set
def load_MNIST(data_path, validation, test_split):
    num_channels = 1 # Monocolour images
    image_size = 28 # 28x28 images
    num_classes = 10 # characters 0-9

    # 55,000 training, 10,000 test, 5,000 validation
    # if not present in data location, try to download
    mnist = input_data.read_data_sets(data_path, one_hot=True)

    training_set = mnist.train.images
    training_labels = mnist.train.labels
    testing_set = mnist.test.images
    testing_labels = mnist.test.labels

    # reshape data 
    training_set = training_set.reshape(-1, image_size, image_size, num_channels).astype(np.float32)
    testing_set = testing_set.reshape(-1, image_size, image_size, num_channels).astype(np.float32)

    # if validation set is required, partition segment of test data
    if validation == True:
        testing_set, testing_labels, validation_set, validation_labels = create_validation_set(testing_set, testing_labels, test_split)
    # otherwise just create an empty array
    else: validation_set, validation_labels = np.empty(shape=(2, 2))
        
    return training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels, image_size, num_channels, num_classes

# function that creates validation set from test set and supplied ration
#   @param testing_set = numpy array containing testing images
#   @param testing_labels = numpy array that contains testing labels
#   @param test_split = int value that states percentage to use for testing set
def create_validation_set(testing_set, testing_labels, test_split):
    rows = testing_set.shape[0]
    valid_start = int(rows * test_split/100)

    test_set = testing_set[:valid_start]
    valid_set = testing_set[valid_start:]

    test_labels = testing_labels[:valid_start]
    valid_labels = testing_labels[valid_start:]

    # garbage collect these
    del testing_set
    del testing_labels

    # return reduced test set and validation set + labels
    return test_set, test_labels, valid_set, valid_labels