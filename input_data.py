from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pickle

def load_prima_head_pose():

    image_size = 64 # images are 32x32x3
    num_channels = 3 # RGB
    num_classes = 10 # 10 possible classes

    pickle_directory = "./data/prima_pitch_p1_out.pickle"
    
    with open(pickle_directory, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
        training_data = data[b'training_dataset']
        training_labels = data[b'training_label']
        testing_data = data[b'validation_dataset']
        testing_labels = data[b'validation_label']
    
    print(training_data.shape)
    print(training_labels.shape)
    print(testing_data.shape)
    print(testing_labels.shape)

    training_data = training_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    training_labels = training_labels.reshape((-1, 1)).astype(np.float32)

    testing_data = testing_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    testing_labels = testing_labels.reshape((-1, 1)).astype(np.float32)

    print(training_data.shape)
    print(training_labels.shape)
    print(testing_data.shape)
    print(testing_labels.shape)

    return training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes


# function to load data set and parameters for CIFAR10
def load_CIFAR_10(validation=True):
    num_channels = 3 # RGB
    image_size = 32 # 32x32 images
    num_classes = 10 # 10 possible classes. info @ https://www.cs.toronto.edu/~kriz/cifar.html
    pickle_directory = "C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/data/CIFAR-10/cifar-10-batches-py/" # DONT WANT THIS TO BE HARDCODED
    num_training_files = 5 # CIFAR10 training data is split into 5 files
    num_images_per_file = 10000 
    num_training_images_total = num_training_files * num_images_per_file
    num_testing_images_total = 10000

    # training images
    training_set = np.zeros(shape=[num_training_images_total, image_size, image_size, num_channels], dtype=float)
    # training class numbers as integers
    training_classes = np.zeros(shape=[num_training_images_total], dtype=int)
    # training class labels in one hot encoding
    training_labels = np.zeros(shape=[num_training_images_total, num_classes], dtype=int)

    begin = 0

    for i in range(num_training_files):
        pickle_file = pickle_directory + "data_batch_" + str(i + 1)
            
        with open(pickle_file, mode='rb') as file:
            data = pickle.load(file, encoding='bytes')
            images_batch = data[b'data']
            classes_batch = np.array(data[b'labels'])

            images_batch = images_batch.reshape([-1, num_channels, image_size, image_size])
            images_batch = images_batch.transpose([0, 2, 3, 1])

            num_images = len(images_batch)
            end = begin + num_images

            training_set[begin:end, :] = images_batch
            training_classes[begin:end] = classes_batch

            begin = end

    # convert training labels from integer format to one hot encoding
    training_labels = np.eye(num_classes, dtype=int)[training_classes]
            
    # testing class labels in one hot encoding
    testing_labels = np.zeros(shape=[num_testing_images_total, num_classes], dtype=int)

    pickle_file = pickle_directory + "test_batch"
    with open(pickle_file, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
        testing_set = data[b'data']
        testing_classes = np.array(data[b'labels'])

        testing_set = testing_set.reshape([-1, num_channels, image_size, image_size])
        testing_set = testing_set.transpose([0, 2, 3, 1])

        del data

    # convert testing set labels from integer format to one hot encoding
    testing_labels = np.eye(num_classes, dtype=int)[testing_classes]

    # reshape data 
    training_set = training_set.reshape(-1, image_size, image_size, num_channels).astype(np.float32)
    testing_set = testing_set.reshape(-1, image_size, image_size, num_channels).astype(np.float32)

    if validation == True:
        testing_set, testing_labels, validation_set, validation_labels = create_validation_set(testing_set, testing_labels)
    else: validation_set, validation_labels = np.empty(shape=(2, 2))
        
    return training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels, image_size, num_channels, num_classes


# function to loead data set and parameters for MNIST
def load_MNIST(validation=False):
    num_channels = 1 # Monocolour images
    image_size = 28 # 28x28 images
    num_classes = 10 # characters 0-9

    # 55,000 training, 10,000 test, 5,000 validation
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) # DONT WANT THIS TO BE HARDCODED

    training_set = mnist.train.images
    training_labels = mnist.train.labels
    testing_set = mnist.test.images
    testing_labels = mnist.test.labels

    training_set = training_set.reshape(-1, image_size, image_size, num_channels).astype(np.float32)
    testing_set = testing_set.reshape(-1, image_size, image_size, num_channels).astype(np.float32)

    if validation == True:
        testing_set, testing_labels, validation_set, validation_labels = create_validation_set(testing_set, testing_labels)
    else: validation_set, validation_labels = np.empty(shape=(2, 2))
        
    return training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels, image_size, num_channels, num_classes


def create_validation_set(testing_set, testing_labels):
    testing_set, validation_set = np.split(testing_set, 2)
    testing_labels, validation_labels = np.split(testing_labels, 2)
    return testing_set, testing_labels, validation_set, validation_labels