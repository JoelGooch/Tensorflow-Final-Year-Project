import tensorflow as tf

# factory used to create relveant layer dependent upon input
class layer_factory():

    def __init__(self, dataSet, allLayers):
        self.currConvLayer = 0
        self.numConvLayersTotal = 0
        self.numDenseLayers = 0
        self.newLayers = []
        self.dataSet = dataSet
        self.allLayers = allLayers
        print(self.dataSet)

    def createLayer(self, inputLayer, layer):
        if layer.layerType == 'Convolution':
            newLayer = self.new_conv_layer(inputLayer, layer.layerName, layer.kernelSize, layer.stride, layer.actFunction, layer.numOutputFilters, layer.padding, layer.normalize, layer.dropout, layer.keepRate)
            self.numConvLayersTotal += 1
        elif layer.layerType == 'Max Pool':
            newLayer = self.new_max_pool_layer(inputLayer, layer.layerName, layer.kernelSize, layer.stride, layer.padding, layer.normalize, layer.dropout, layer.keepRate)
        elif layer.layerType == 'Dense':
            # if this is the first dense layer, reshape its incoming layer
            if self.numDenseLayers == 0:
                inputLayer = self.flatten_layer(inputLayer)
            newLayer = self.new_dense_layer(inputLayer, layer.layerName, layer.numOutputNodes, layer.actFunction, layer.normalize, layer.dropout, layer.keepRate)
            self.numDenseLayers += 1
        elif layer.layerType == 'Output':
            newLayer = self.new_output_layer(inputLayer, layer.layerName, layer.actFunction)

        self.newLayers.append(newLayer)
        return newLayer

    # helper function to create weights
    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    # helper function to create biases
    def new_biases(self, length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    # helper function to create a convolution layer
    def new_conv_layer(self, inputLayer, layerName, filter_size, stride, act_function, num_output_filters, padding, normalize, dropout, keepRate):

        self.currConvLayer = 0

        if not self.newLayers:
            if self.dataSet == 'CIFAR10':
                num_input_channels = 3
            elif self.dataSet == 'MNIST':
                num_input_channels = 1
        else:
            for e in self.allLayers:
                if e.layerType == 'Convolution':
                    self.currConvLayer += 1
                    if self.currConvLayer == self.numConvLayersTotal:
                        num_input_channels = e.numOutputFilters
                        break

        print("conv inputs {0}".format(num_input_channels))
        print("conv outputs {0}".format(num_output_filters))

        shape = [filter_size, filter_size, num_input_channels, num_output_filters]

        weights = self.new_weights(shape=shape)
        biases = self.new_biases(length=num_output_filters)

        layer = tf.nn.conv2d(input=inputLayer, filter=weights, strides=[1, stride, stride, 1], padding=padding)
        layer += biases

        if act_function == 'ReLu':
            layer = tf.nn.relu(layer)
        elif act_funcion == 'Sigmoid':
            layer = tf.sigmoid(layer)
        if normalize == 'True':
            layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        if dropout == 'True':
            layer = tf.nn.dropout(layer, keepRate)
        return layer

        # helper function to create a new max pooling layer
    def new_max_pool_layer(self, inputLayer, layerName, kernel_size, stride, padding, normalize, dropout, keep_rate):
        layer = tf.nn.max_pool(inputLayer, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding)
        if normalize == 'True':
            layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        if dropout == 'True':
            layer = tf.nn.dropout(layer, keepRate)
        print("pool")
        
        return layer

    # helper function flatten layer, ready to feed into dense layer
    def flatten_layer(self, layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat

    # helper function to create a dense/fully connected layer
    def new_dense_layer(self, inputLayer, layerName, num_outputs, act_function, normalize, dropout, keepRate):

        currDenseLayers = 0

        print("next dense")

        if self.numDenseLayers == 0:
            if self.dataSet == 'CIFAR10':
                image_size = 32
                print("cifar")
            if self.dataSet == 'MNIST':
                image_size = 28
                print("mnist")
            if self.dataSet == 'Prima head pose':
                image_size = 64

            for e in self.allLayers:
                if e.layerType == 'Max Pool':
                    image_size /= e.stride


            for e in self.allLayers[::-1]:
                if e.layerType == 'Convolution':
                    prev_filter_size = e.numOutputFilters
                    print("prev filter size ", prev_filter_size)
                    break

            print("img size {0} * img size {1} * prev filter size {2}".format(image_size, image_size, prev_filter_size))
            num_inputs = image_size * image_size * prev_filter_size

        else:
            for e in self.allLayers:
                if e.layerType == 'Dense':
                    currDenseLayers += 1
                    if currDenseLayers == self.numDenseLayers:
                        num_inputs = e.numOutputNodes
                        break

        print("dense inputs {0}".format(num_inputs))
        print("dense outputs {0}".format(num_outputs))

        weights = self.new_weights(shape=[int(num_inputs), num_outputs])
        biases = self.new_biases(length=num_outputs)

        layer = tf.matmul(inputLayer, weights) + biases
        if act_function == 'ReLu':
            layer = tf.nn.relu(layer)
        elif act_funcion == 'Sigmoid':
            layer = tf.sigmoid(layer)
        if normalize == 'True':
            layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        if dropout == 'True':
            layer = tf.nn.dropout(layer, keepRate)

        print("finished dense")
        return layer

    # helper function to create an output layer
    def new_output_layer(self, inputLayer, layerName, act_function):
        
        num_inputs = self.allLayers[-2].numOutputNodes
        print("output inputs {0}".format(num_inputs))

        if self.dataSet == 'CIFAR10' or self.dataSet == 'MNIST':
            num_outputs = 10

        print("output outputs {0}".format(num_outputs))


        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)
        layer = tf.matmul(inputLayer, weights) + biases
        if act_function == 'ReLu':
            layer = tf.nn.relu(layer)
        elif act_function == 'Sigmoid':
            layer = tf.sigmoid(layer)
        return layer