import tensorflow as tf
import numpy as np

# factory pattern implementation used to create relevant layer at runtime depending on input
class layer_factory():

    def __init__(self, num_classes, num_channels, all_layers):
        # references storing current number of conv and dense layers
        self.num_conv_layers = 0
        self.num_FC_layers = 0
        # list to contain newly created layers
        self.new_layers = []
        # general params
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.all_layers = all_layers

    # user to create a single layer, which layer is created is dependent upon input
    #   @param input_layer = the previous Tensorflow layer
    #   @param layer = layer to be created in the Layer class format parsed from .XML, will be converted to a Tensorflow computation layer
    #   @param input_dimension = dimension of previous layers outputs
    def create_layer(self, input_layer, layer, input_dimension):  

        if layer.layer_type == 'Convolution':
            # create new convolution layer from user defined params
            new_layer, num_outputs = self.new_conv_layer(input_layer, input_dimension, layer.layer_name, layer.kernel_size, layer.stride, layer.act_function, 
                layer.num_output_filters, layer.weight_init, layer.weight_val, layer.bias_init, layer.bias_val, layer.padding, layer.normalize, layer.dropout, 
                layer.keep_rate)
            # increment number of convolution layers present
            self.num_conv_layers += 1

        elif layer.layer_type == 'Max Pool':
            # create new max pooling layer from user defined params
            new_layer, num_outputs = self.new_max_pool_layer(input_layer, input_dimension, layer.kernel_size, layer.stride, layer.padding, layer.normalize, 
                layer.dropout, layer.keep_rate)

        elif layer.layer_type == 'Fully Connected':
            # if this is the first fully connected  layer, reshape its incoming layer
            if self.num_FC_layers == 0:
                input_layer = self.flatten_layer(input_layer)
            # create new fully connected layer from user defined params
            new_layer, num_outputs = self.new_fully_connected_layer(input_layer, input_dimension, layer.layer_name, layer.num_output_nodes, layer.weight_init, 
                layer.weight_val, layer.bias_init, layer.bias_val, layer.act_function, layer.dropout, layer.keep_rate)
            # increment number of fully connected layers present
            self.num_FC_layers += 1

        elif layer.layer_type == 'Output':
            # create new output layer from user defined params
            new_layer, num_outputs = self.new_output_layer(input_layer, input_dimension, layer.layer_name, layer.act_function, layer.weight_init, layer.weight_val, 
                layer.bias_init, layer.bias_val)

        # append to new layers array
        self.new_layers.append(new_layer)

        # return layer and number of outputs to be used as number of inputs to next layer
        return new_layer, num_outputs

    # helper function to create weights defined by user
    #   @param shape = shape of weights to be created
    #   @param name = string name of weights
    #   @param conv = bool stating if the weights are for a convolution layer
    #   @param init = method of initialization to be used
    #   @param val = standard deviation value to be used
    def new_weights(self, shape, name, conv, init, val):

        if conv == True:
            weight_name = name + "_Cweights" # for convolution weights
        else: weight_name = name + "_Dweights" # for dense weights
        
        if init == 'Random Normal':
            return tf.Variable(tf.random_normal(shape=shape, stddev=val), name=weight_name)
        elif init == 'Truncated Normal':
            return tf.Variable(tf.truncated_normal(shape=shape, stddev=val), name=weight_name)
        elif init == 'Xavier':
            return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

    # helper function to create biases defined by user
    #   @param length = number of bias values to be created
    #   @param name = string name of weights
    #   @param init = method of initialization to be used
    #   @param val = standard deviation value to be used
    def new_biases(self, length, name, init, val):

        bias_name = name + "_biases"
        if init == 'Random Normal':
            return tf.Variable(tf.random_normal(shape=[length], stddev=val), name=bias_name)
        elif init == 'Truncated Normal':
            return tf.Variable(tf.truncated_normal(shape=[length], stddev=val), name=bias_name)
        elif init == 'Zeros':
            return tf.Variable(tf.constant(val, shape=[length]), name=bias_name)
        elif init == 'Constant':
            return tf.Variable(tf.zeros(shape=[length]), name=bias_name)

    # helper function to create a convolution layer
    #   @param input_layer = the previous Tensorflow layer
    #   @param input_dimension = dimension of previous layers outputs
    #   @param layer_name = string name of layer to create
    #   @param filter_size = integer size of convolution filter to used
    #   @param act_function = string name of activation function to use
    #   @param num_output_filters = integer number of output filters to create
    #   @param weight_init = method of weight initialization to be used
    #   @param weight_val = standard deviation value of weights to be used
    #   @param bias_init = method of bias initialization to be used
    #   @param bias_val = standard deviation/constant value of biases to be used
    #   @param padding = string containing form of padding to use
    #   @param normalize = bool if normalization is used on layer
    #   @param dropout = bool if dropout is used on layer
    #   @param keep_rate = float value of dropout keep rate
    def new_conv_layer(self, input_layer, input_dimension, layer_name, filter_size, stride, act_function, num_output_filters, weight_init, weight_val,
        bias_init, bias_val, padding, normalize, dropout, keep_rate):

        # if this is first convolution layer, number of input channels is num channels of image
        if not self.new_layers:
            num_input_channels = self.num_channels
        # otherwise, locate the most recently made convolution layer and use its number of output filters value
        else:
            curr_conv_layer = 0
            # cycle all layers in full list of layers
            for e in self.all_layers:
                if e.layer_type == 'Convolution':
                    curr_conv_layer += 1
                    # if this layer is the last one created so far 
                    if curr_conv_layer == self.num_conv_layers:
                        # use its number of output filters as number of channels for new layer
                        num_input_channels = e.num_output_filters
                        break

        # construct shape from user parameters
        shape = [filter_size, filter_size, num_input_channels, num_output_filters]

        # create layer weights and biases
        weights = self.new_weights(shape=shape, name=layer_name, conv=True, init=weight_init, val=weight_val)
        biases = self.new_biases(length=num_output_filters, name=layer_name, init=bias_init, val=bias_val)

        # create tensorflow 2d convolution call using obtained parameters
        layer = tf.nn.conv2d(input=input_layer, filter=weights, strides=[1, stride, stride, 1], padding=padding) + biases

        # calculate new dimension of input image depending on PADDING and STRIDE selected
        num_outputs = self.calc_output_dimension(input_dimension, filter_size, padding, stride)

        # select which activation function is present (if any)
        layer = self.select_activation_function(layer, act_function)

        # add normalization if defined by user
        layer = self.select_normalize(layer, normalize)

        # add dropout if defined by user
        layer = self.select_dropout(layer, dropout, keep_rate)

        return layer, num_outputs

    # helper function to create a new max pooling layer
    #   @param input_layer = the previous Tensorflow layer
    #   @param input_dimension = dimension of previous layers outputs
    #   @param kernel_name = integer size of pooling kernel to use
    #   @param stride = integer size of stride to use during pooling
    #   @param padding = string containing form of padding to use
    #   @param normalize = bool if normalization is used on layer
    #   @param dropout = bool if dropout is used on layer
    #   @param keep_rate = float value of dropout keep rate
    def new_max_pool_layer(self, input_layer, input_dimension, kernel_size, stride, padding, normalize, dropout, keep_rate):

        # add max pooling operator
        layer = tf.nn.max_pool(input_layer, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding)

        # calculate new dimension of input image depending on PADDING and STRIDE selected
        num_outputs = self.calc_output_dimension(input_dimension, kernel_size, padding, stride)

        # add normalization if defined by user
        layer = self.select_normalize(layer, normalize)

        # add dropout if defined by user
        layer = self.select_dropout(layer, dropout, keep_rate)
        
        return layer, num_outputs

    # helper function flatten layer, ready to feed into fully connected  layer
    #   @param layer = the Tensorflow layer to flatten
    def flatten_layer(self, layer):

        # get current shape of layer
        layer_shape = layer.get_shape()

        # extract number of features from shape
        num_features = layer_shape[1:4].num_elements()

        # reshape layer using number of features
        layer_flat = tf.reshape(layer, [-1, num_features])

        return layer_flat

    # helper func-tion to create a fully connected layer
    #   @param input_layer = the previous Tensorflow layer
    #   @param input_dimension = dimension of previous layers outputs
    #   @param layer_name = string name of layer to create
    #   @param num_outputs = number of output nodes of layer
    #   @param weight_init = method of weight initialization to be used
    #   @param weight_val = standard deviation value of weights to be used
    #   @param bias_init = method of bias initialization to be used
    #   @param bias_val = standard deviation/constant value of biases to be used
    #   @param act_function = string name of activation function to use
    #   @param dropout = bool if dropout is used on layer
    #   @param keep_rate = float value of dropout keep rate
    def new_fully_connected_layer(self, input_layer, input_dimension, layer_name, num_outputs, weight_init, weight_val, bias_init, bias_val, 
        act_function, dropout, keep_rate):

        # if this is the first fully connected  layer in the network
        if self.num_FC_layers == 0:
            # cycle all layers in reverse until finding the first (last in choronological order) convolution layer present
            for e in self.all_layers[::-1]:
                if e.layer_type == 'Convolution':
                    # take note of how many output filters were present
                    prev_filter_size = e.num_output_filters
                    break

            # calculate the dimension of image 
            num_inputs = input_dimension * input_dimension * prev_filter_size
        else:
            num_inputs = input_dimension

        # create layer weights and biases
        weights = self.new_weights(shape=[int(num_inputs), num_outputs], conv=False, name=layer_name, init=weight_init, val=weight_val)
        biases = self.new_biases(length=num_outputs, name=layer_name, init=bias_init, val=bias_val)

        # perform weighted summation
        layer = tf.matmul(input_layer, weights) + biases

        # select which activation function is present (if any)
        layer = self.select_activation_function(layer, act_function)

        # add dropout if defined by user
        layer = self.select_dropout(layer, dropout, keep_rate)

        return layer, num_outputs

    # helper function to create an output layer
    #   @param input_layer = the previous Tensorflow layer
    #   @param input_dimension = dimension of previous layers outputs
    #   @param layer_name = string name of layer to create
    #   @param act_function = string name of activation function to use
    #   @param weight_init = method of weight initialization to be used
    #   @param weight_val = standard deviation value of weights to be used
    #   @param bias_init = method of bias initialization to be used
    #   @param bias_val = standard deviation/constant value of biases to be used
    def new_output_layer(self, input_layer, input_dimension, layer_name, act_function, weight_init, weight_val, bias_init, bias_val):
        
        # take number of inputs from second to last layer in all layers array
        num_inputs = self.all_layers[-2].num_output_nodes

        # create layer weights and biases
        weights = self.new_weights(shape=[num_inputs, self.num_classes], conv=False, name=layer_name, init=weight_init, val=weight_val)
        biases = self.new_biases(length=self.num_classes, name=layer_name, init=bias_init, val=bias_val)

        layer = tf.matmul(input_layer, weights) + biases

        layer = self.select_activation_function(layer, act_function)

        return layer, input_dimension

    # function that calculates the output image dimension from its incoming parameters
    #   @param input_dimension = dimension of previous layers outputs
    #   @param filter size = integer value of filter size of current layer
    #   @param padding = string value of which form of padding is used in current layer
    #   @param stride = integer value of which stride value is used on current layer
    def calc_output_dimension(self, input_dimension, filter_size, padding, stride):
        if padding == 'SAME':
            num_outputs = np.ceil(float(input_dimension) / float(stride))
        elif padding == 'VALID':
            num_outputs = np.ceil(float(input_dimension - filter_size + 1) / float(stride))
        return num_outputs

    # function that adds activation function, if user requires
    #   @param layer = current Tensorflow layer
    #   @param act_function = string name of activation function to use on current layer
    def select_activation_function(self, layer, act_function):
        if act_function == 'ReLu':
            layer = tf.nn.relu(layer)
        elif act_function == 'Sigmoid':
            layer = tf.sigmoid(layer)

        return layer

    # function that adds normalization, if user requires
    #   @param layer = current Tensorflow layer
    #   @param normalize = bool if normalize is used on current layer
    def select_normalize(self, layer, normalize):
        if normalize == True:
            layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        return layer

    # function that adds dropout, if user requires
    #   @param layer = current Tensorflow layer
    #   @param dropout = bool if dropout is used on layer
    #   @param keep_rate = float value of dropout keep rate
    def select_dropout(self, layer, dropout, keep_rate):
        if dropout == True:
            layer = tf.nn.dropout(layer, keep_rate)    
        return layer   