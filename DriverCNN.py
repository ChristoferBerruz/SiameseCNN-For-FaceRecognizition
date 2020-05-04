import tensorflow.compat.v1 as tf
import os, sys
import cv2
from tensorflow.keras.utils import to_categorical
from DenseSiameseCNN import DenseSiameseCNN
import numpy as np
import random


width = 90
height = 90
def load_data():
    global width, height

    sample_number_testing =  200
    sample_number_training = 200
    train = np.empty((sample_number_training, height, width), dtype = 'float64')
    test = np.empty((sample_number_testing, height, width), dtype = 'float64')
    label_test = np.zeros((sample_number_testing, 1), dtype = int)
    label_training = np.zeros((sample_number_training, 1), dtype = int)
    #Loading the images
    i = 0
    for filename in os.listdir('C:/Users/anast/Documents/Computer-Vision/AttDataSet/ATTDataSet/Training'):
        category, sample_index = filename.split('_')
        #the category are always Scategory, and category might have more than 1 digit
        size_category = len(category)
        label = int(category[1:size_category])-1 #Retrieving the label from the test
        label_training[i] = label
        temp = cv2.imread('C:/Users/anast/Documents/Computer-Vision/AttDataSet/ATTDataSet/Training/{0}'.format(filename), 0)/255.0 
        train[i] = cv2.resize(temp, (width, height))
        i += 1

    i = 0
    for filename in os.listdir('C:/Users/anast/Documents/Computer-Vision/AttDataSet/ATTDataSet/Testing'):
        category, sample_indx = filename.split('_')
        size_category = len(category)
        label = int(category[1:size_category])-1
        label_test[i] = label
        temp = cv2.imread('C:/Users/anast/Documents/Computer-Vision/AttDataSet/ATTDataSet/Testing/{0}'.format(filename), 0)/255.0
        test[i] = cv2.resize(temp, (width, height))
        i += 1
    train = np.reshape(train, (train.shape[0], train.shape[1]*train.shape[2]))
    test = np.reshape(test, (test.shape[0], test.shape[1]*test.shape[2]))
    training_augmented = np.hstack((train, label_test))
    testing_augmented = np.hstack((test, label_test))
    return train,training_augmented,  test, testing_augmented, label_training, label_test

def create_pairs(x, digit_indices, num_classes):
    """
    Creates pairs of data
    """
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def main():
    global width, height
    tf.disable_v2_behavior()
    num_classes = 40
    trainingX, training_augmented, testingX, testing_augmented, label_training, label_testing = load_data()
    SiameseCNN = DenseSiameseCNN(height, width, 1, num_classes)
    increase_training = False #If we want to feed extra data
    if increase_training:
        extra = 120 #Adding 120 samples for training, so we are using 80% for training
        trainingX = np.vstack((trainingX, testingX[0:extra]))
        label_training = np.vstack((label_testing, label_testing[0:extra]))
        testingX = testingX[extra:]
        label_testing = label_testing[extra:]
        training_augmented = np.hstack((trainingX, label_training))
    num_iterations = 500
    batch_size = 20
    SiameseCNN.trainSiamese(Data_augmented = training_augmented, num_iterations = num_iterations, batch_size = batch_size)
    SiameseCNN.trainSiameseForClassification(training_augmented, num_iterations = num_iterations, batch_size = batch_size)
    SiameseCNN.compute_accuracy(testingX, label_testing)
    """
        SiameseCNN = DenseSiameseCNN(input_height = 112, input_width=92, input_channels = 1)
        # create training+test positive and negative pairs
        digit_indices = [np.where(label_training == i)[0] for i in range(num_classes)]
        training_pairs, training_y = create_pairs(trainingX, digit_indices, num_classes)

        digit_indices = [np.where(label_testing == i)[0] for i in range(num_classes)]
        testing_pairs, testing_y = create_pairs(testingX, digit_indices, num_classes)

        first_data = training_pairs[:,0]
        second_data = training_pairs[:,1]

        epochs = 5
        batch_size = 10
        SiameseCNN.train_siamese(first_data, second_data, np.array(training_y, dtype = 'float32'),batch_size = batch_size, epochs = epochs)

        y_one_hot = to_categorical(label_training, num_classes)
        SiameseCNN.train_dense_network(trainingX, y_one_hot)
    """

if __name__ == '__main__':main()