import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
import os
import pickle


class DenseSiameseCNN(object):
    """
    Convolutional CNN network followed by a classifier. It expects the input images to be in format
    [image_height, image_width, number_of_channels]. Please note that we expect the image
    to be a square image, pad the image if necessary
    """

    def __init__(self, input_height, input_width, input_channels, number_of_categories):
        self.tf_input_first = tf.placeholder(tf.float32, [None, input_height, input_width, input_channels], name = 'input_first')
        self.tf_input_second = tf.placeholder(tf.float32, [None, input_height, input_width, input_channels], name = 'input_second')
        self._input_width = input_width
        self._input_height = input_height
        self._number_of_categories = number_of_categories
        #Labels fot the image pair. I.e. for 100 pairs of images, 100 labels
        # 1 : Similar, 0 : dissimilar
        self.tf_Y = tf.placeholder(tf.float32, [None, ], name = 'Y')
        self.tf_Y_one_hot = tf.placeholder(tf.float32, [None, number_of_categories], name = 'Y_one_hot')
        self.number_of_categories = number_of_categories

        #Outputs, loss functions, and training optimizer
        self.tf_output_first, self.tf_output_second = self.siameseNetwork()
        self.output = self.siameseNetworkWithClassification()
        self.loss = self.contrastiveLoss()
        self.loss_dense_network = self.crossEntropyLoss()
        self.optimizer = self.optimizer_init()
        self.optimizer_crossEntropy  = self.optimizer_init_crossEntropy()
        self.saver = tf.train.Saver()

        #Initialize tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def layer(self, tf_input, num_hidden_units, variable_name, trainable = True):
        """
        Simple dense, fully connected layer
        Inputs:
            - tf_input : a tensor of dimensions batch_sizexfeatures (32x784 for example)
            - num_hidden_units : number of hidden units
        Outputs:
            - out : tf_input@W + b
        """

        tf_weight_initializer = tf.random_normal_initializer(mean = 0, stddev = 0.01)
        num_features = tf_input.get_shape()[1]
        W = tf.get_variable(
            name = variable_name + '_W',
            dtype = tf.float32, 
            shape = [num_features, num_hidden_units],
            initializer = tf_weight_initializer,
            trainable = trainable
            )
        b = tf.get_variable(
            name = variable_name + '_b',
            dtype = tf.float32,
            shape = [num_hidden_units],
            trainable = trainable
            )
        out = tf.add(tf.matmul(tf_input, W),b)
        return out

    def network(self, tf_input, trainable = True):
        """
        Fully connected neural net to produce an embedding/dimensionality reduction
        Inputs:
            - tf_input : tf.tensor that is the input of the neural net
        Outputs:
            - tf.tensor represeting 2D embedding of the input
        """
        with tf.variable_scope('siamese'):
            #First layer
            conv1 = tf.layers.Conv2D(filters = 4, kernel_size = 7, strides = 2, activation = tf.nn.tanh, trainable = trainable)(tf_input)
            pool1 = tf.layers.AveragePooling2D(pool_size = 3, strides = 2)(conv1)
            #Second layer
            conv2 = tf.layers.Conv2D(filters = 8, kernel_size = 5, activation = tf.nn.relu, trainable = trainable)(pool1)
            pool2 = tf.layers.AveragePooling2D(pool_size = 3, strides = 2)(conv2)
            #Third layer
            conv3 = tf.layers.Conv2D(filters = 16, kernel_size = 3, trainable = trainable)(pool2)
            pool3 = tf.layers.MaxPooling2D(pool_size = 2, strides = 2)(conv3)
            output = tf.layers.Flatten()(pool3)
            return output


    def network_with_classification(self, tf_input):
        """
        Creates a neural network that uses the siamese network, and then forward
        passes the input to a dense network that does classification
        Inputs:
            - tf_input : input to be classified

        Returns:
            layer that is supposed to do classification
        """
        #Calling the network to do embedding, but we no longer want to train the embedder.
        s3 = self.network(tf_input = tf_input, trainable = False)
        a3 = s3
        #Now we are forward passing to the classifier network and we want to train it!
        s4 = self.layer(tf_input = a3, num_hidden_units = 480, variable_name = 's4', trainable = True) 
        a4 = tf.nn.relu(s4)
        s5 = self.layer(tf_input = a4, num_hidden_units = self.number_of_categories, variable_name = 's5', trainable = True)
        return s5


    def siameseNetwork(self):
        """
        Forward pass of the pair of data
        Inputs:
            - None : pair of data must be a property of this class
        Returns:
            - both outputs of the corresponding pair
        """
        with tf.variable_scope("siamese") as scope:
            output_first = self.network(self.tf_input_first) #First datum forward pass
            scope.reuse_variables() #Make sure we use the same parameters
            output_second = self.network(self.tf_input_second) #Second datum forward pass
        return output_first, output_second

    def siameseNetworkWithClassification(self):
        """
        Builds a siamese network with a classifier attached to it
        """
        with tf.variable_scope("siamese", reuse = tf.AUTO_REUSE) as scope:
            output = self.network_with_classification(self.tf_input_first)
        return output

    def contrastiveLoss(self, margin = 5.0):
        """
        Contrastive loss as specified by LeCun.
        Inputs:
            - margin : float, margin for the loss
        Returns:
            mean constrative loss of the batch
        """

        with tf.variable_scope('siamese') as scope:
            labels = self.tf_Y
            #The old fashion Dw term
            dist = tf.pow(tf.subtract(self.tf_output_first, self.tf_output_second), 2, name = 'Dw')
            Dw = tf.reduce_sum(dist, 1) #Summing over all xi of the vector
            #We add a small epsilon to stabilize gradients
            Dw2 = tf.sqrt(Dw + 1e-6, name = 'Dw2')

            #Loss functions
            lossSimilar = tf.multiply(labels, tf.pow(Dw2, 2), name = 'constrative_loss_sim')
            lossDissimilar = tf.multiply(tf.subtract(1.0, labels), tf.pow(tf.maximum(tf.subtract(margin, Dw2), 0), 2),name = 'constrative_loss_dissim')
            loss = tf.reduce_mean(tf.add(lossSimilar, lossDissimilar), name = 'constrative_loss')
            return loss

    def crossEntropyLoss(self):
        labels = self.tf_Y_one_hot
        lossd = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.output, labels = labels))
        return lossd

    def optimizer_init(self):
        ALPHA = 0.01
        RAND_SEED = 0
        tf.set_random_seed(RAND_SEED)
        optimizer = tf.train.GradientDescentOptimizer(ALPHA).minimize(self.loss)
        return optimizer

    def optimizer_init_crossEntropy(self):
        ALPHA = 0.01
        RAND_SEED = 0
        tf.set_random_seed(RAND_SEED)
        optimizer = tf.train.AdamOptimizer(ALPHA).minimize(self.loss_dense_network)
        return optimizer

    def trainSiamese(self, Data_augmented, num_iterations, batch_size = 32):
        """
        Trains the siamese in a given number of iterations, or epochs. Data is passed as pairs
        Inputs:
            - first_data (ndarray) : contains the images to pass first 
            - second_data (ndarray) : contains the images to pass second
            - labels (ndarray) : labels for the pair
            - num_iterations (int): number of epochs
            - batch_size : batch size to be fed
        """
        last_col_idx = Data_augmented.shape[1] - 1
        M = Data_augmented.shape[0]
        trainingLoss = 0.0
        for epoch in range(num_iterations):
            np.random.shuffle(Data_augmented)
            X_data = Data_augmented[:, 0:last_col_idx]
            Y_data = Data_augmented[:, last_col_idx]
            for i in range(0, M, 2*batch_size):
                (input_first_D, y1)= X_data[i:i+batch_size], Y_data[i:i+batch_size]
                (input_second_D, y2) = X_data[i+batch_size:i+2*batch_size], Y_data[i+batch_size:i+2*batch_size]
                input_first_D = np.reshape(input_first_D, (batch_size, self._input_height, self._input_width, 1))
                input_second_D = np.reshape(input_second_D, (batch_size, self._input_height, self._input_width, 1))
                label = np.where(y1==y2, 1.0, 0.0)
                #print(label)
                _, trainingLoss = self.sess.run([self.optimizer, self.loss],feed_dict = {self.tf_input_first:input_first_D,
                                                                                        self.tf_input_second:input_second_D,
                                                                                        self.tf_Y:label}
                                                )
            print('Epoch = %d, Loss = %.4E' % (epoch, trainingLoss))

    
    
    def trainSiameseForClassification(self, Data_augmented, num_iterations, batch_size = 10):
        """
        Trains the siamese network with the classifier
        """
        last_col_idx = Data_augmented.shape[1] - 1
        M = Data_augmented.shape[0]
        trainingloss = 0.0
        for epoch in range(num_iterations):
            np.random.shuffle(Data_augmented)
            for i in range(0, M, batch_size):
                input1 = Data_augmented[i:i+batch_size, 0:last_col_idx]
                input1 = np.reshape(input1, (batch_size, self._input_height, self._input_width, 1))
                y1 = Data_augmented[i:i+batch_size, last_col_idx]
                one_hot = to_categorical(y1, self._number_of_categories) #convert labels to one-hot encoding
                labels = np.zeros(batch_size)
                _, trainingloss = self.sess.run([self.optimizer_crossEntropy, self.loss_dense_network],
                                                feed_dict = {self.tf_input_first:input1, self.tf_input_second:input1,
                                                             self.tf_Y_one_hot:one_hot, self.tf_Y:labels}
                                                )
            print("Epoch = %d, Loss = %.4E" % (epoch, trainingloss))


    def test_model(self, input):
        input = np.reshape(input, (input.shape[0], self._input_height, self._input_width, 1))
        output = self.sess.run(self.tf_output_first, feed_dict = {self.tf_input_first:input})
        return output

    
    def save_model(self, file_path_absolute, model_name):
        if not os.path.exists(file_path_absolute):
            os.makedirs(file_path_absolute)
        self.saver.save(self.sess, file_path_absolute + model_name)

    def save_weights(self, w1, b1, w2, b2):
        weights = {'w1':w1, 'b1':b1,
                   'w2':w2, 'b2':b2}
        with open('savedWeights.pickle', 'wb') as handle:
            pickle.dump(weights, handle, protocol = pickle.HIGHEST_PROTOCOL)
        print("Your weights have been saved")

    def compute_accuracy(self, testX, labels_test):
        labels = np.zeros(100)
        one_hot = np.zeros((100, self._number_of_categories))
        testX = np.reshape(testX, (testX.shape[0], self._input_height, self._input_width, 1))
        aout = self.sess.run(self.output, feed_dict = {self.tf_input_first:testX,
                                                       self.tf_input_second:testX,
                                                       self.tf_Y_one_hot:one_hot,
                                                       self.tf_Y:labels})
        accuracy_count = 0
        testY = to_categorical(labels_test, self._number_of_categories)
        for i in range(testY.shape[0]):
            max_idx = aout[i].argmax(axis = 0)
            if testY[i, max_idx] == 1:
                accuracy_count += 1
        print("Accuracy: %f %s" % (100.0*accuracy_count/testY.shape[0], "%"))



