import sys
import tensorflow as tf
import numpy as np
import os
import pickle
import datetime
import cv2
import matplotlib
import xml.etree.ElementTree as ET
import CNN_GUI_V10 as design
import input_data as data
import practiceParse as pp
import Layer as l

from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog, QSizePolicy, QListWidgetItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt


# helper function to create weights
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

# helper function to create biases
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

# helper function to create a convolution layer
def new_conv_layer(inputLayer, num_input_channels, filter_size, stride, act_function, num_output_filters, padding, normalize, dropout , keepRate):
    shape = [filter_size, filter_size, num_input_channels, num_output_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_output_filters)
    layer = tf.nn.conv2d(input=inputLayer, filter=weights, strides=[1, stride, stride, 1], padding=padding)
    layer += biases
    if act_function == 'ReLu':
        layer = tf.nn.relu(layer)
    if normalize == 'True':
        layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    if dropout == 'True':
        layer = tf.nn.dropout(layer, keepRate)
    return layer

# helper function flatten layer, ready to feed into dense layer
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat

# helper function to create a dense/fully connected layer
def new_dense_layer(inputLayer, num_inputs, num_outputs, act_function, normalize, dropout, keepRate):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(inputLayer, weights) + biases
    if act_function == 'ReLu':
        layer = tf.nn.relu(layer)
    if normalize == 'True':
        layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    if dropout == 'True':
        layer = tf.nn.dropout(layer, keepRate)
    return layer

# helper function to create an output layer
def new_output_layer(inputLayer, num_inputs, num_outputs, act_function):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(inputLayer, weights) + biases
    if act_function == 'ReLu':
        layer = tf.nn.relu(layer)
    return layer

# helper function to create a new max pooling layer
def new_max_pool_layer(inputLayer, kernel_size, stride, padding, normalize, dropout, keep_rate):
    layer = tf.nn.max_pool(inputLayer, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding)
    if normalize == 'True':
        layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    if dropout == 'True':
        layer = tf.nn.dropout(layer, keepRate)
    return layer

# factory used to create relveant layer dependent upon input
class layer_factory():

    def __init__(self):
        self.numDenseLayers = 0

    def createLayer(self, inputLayer, layer):
        if layer.layerType == 'Convolution':
            newLayer = new_conv_layer(inputLayer, layer.numInputs, layer.kernelSize, layer.stride, layer.actFunction, 
                layer.numOutputFilters, layer.padding, layer.normalize, layer.dropout, layer.keepRate)
        elif layer.layerType == 'Max Pool':
            newLayer = new_max_pool_layer(inputLayer, layer.kernelSize, layer.stride, layer.padding, layer.normalize, layer.dropout, layer.keepRate)
        elif layer.layerType == 'Dense':
            # if this is the first dense layer, reshape input layer
            if self.numDenseLayers == 0:
                inputLayer = flatten_layer(inputLayer)
            newLayer = new_dense_layer(inputLayer, layer.numInputs, layer.numOutputNodes, layer.actFunction, layer.normalize, layer.dropout, layer.keepRate)
            self.numDenseLayers += 1
        elif layer.layerType == 'Output':
            newLayer = new_output_layer(inputLayer, layer.numInputs, layer.numOutputNodes, layer.actFunction)
        return newLayer


class Worker(QObject):

    epochProgress = pyqtSignal(float)
    testSetAccuracy = pyqtSignal(float)
    logMessage = pyqtSignal(str)
    workComplete = pyqtSignal()
    lossOverEpochs = pyqtSignal(float, int)

    def __init__(self, data_set: str, num_epochs: int, batch_size: int, learning_rate: float, optimizer: int, save_directory: str, save_interval: int, model_path: str):
        super().__init__()
        self.end_training = False
        self.data_set = data_set
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.save_directory = save_directory
        self.save_interval = save_interval
        self.model_path = model_path

        
    @pyqtSlot()
    def work(self):
        self.train_network(self.data_set, self.num_epochs, self.batch_size, self.learning_rate, self.optimizer, self.save_directory, self.save_interval, self.model_path)


    def train_network(self, data_set, num_epochs, batch_size, learning_rate, learning_algo, save_path, save_interval, model_path):
        
        # load selected data set
        if (data_set == 'CIFAR10'):
            training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_CIFAR_10()
        if (data_set == 'MNIST'):
            training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_MNIST()

        # normalize data, THIS BREAKS MNIST, CIFAR10 WORKS WITH OR WITHOUT
        #training_set -= 127 
        #testing_set -= 127

        try:
            # get array of layers from XML file
            Layers = pp.getLayers(model_path)
        except:
            self.logMessage.emit('Error reading XML file')
            return 0

        #for layer in Layers:
        #    attrs = vars(layer)
        #    print(', '.join("%s: %s" % item for item in attrs.items()))
        
        graph = tf.Graph()
        with graph.as_default():

            self.logMessage.emit('Initialising Tensorflow Variables... \n')

            # define placeholder variables
            x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
            y = tf.placeholder(tf.float32, shape=(None, num_classes))
            keep_prob = tf.placeholder(tf.float32)

            # stores the class integer values 
            labels_class = tf.argmax(y, dimension=1)
            
            numLayers = len(Layers)
            networkLayers = []

            def CNN_Model(data, _dropout=1.0):

                # instantiate layer factory
                layerFactory = layer_factory()

                # set input to first layer as x (input data)
                layer = x

                # create as many layers as are stored in XML file
                for e in range(len(Layers)):
                    layer = layerFactory.createLayer(layer, Layers[e])
                    networkLayers.append(layer)

                # return last element of layers array (output layer)
                return networkLayers[-1]


            model_output = CNN_Model(x, keep_prob)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model_output, labels=y)

            loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

            loss_summary = tf.summary.scalar("loss", loss)

            global_step = tf.Variable(0, trainable=False)

            #learning_rate = tf.train.exponential_decay(0.0125, global_step, 15000, 0.1, staircase=True)
            #lrate_summary = tf.summary.scalar("learning rate", learning_rate)

            if (learning_algo == 0):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
                self.logMessage.emit('Optimizer: Gradient Descent')
            elif (learning_algo == 1):
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
                self.logMessage.emit('Optimizer: Adam')
            elif (learning_algo == 2):
                optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
                self.logMessage.emit('Optimizer: Ada Grad')
            #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95).minimize(loss, global_step=global_step)

            # convert one hot encoded array to single prediction value
            network_pred_class = tf.argmax(model_output, dimension=1)

            # compare if network prediction was correct
            correct_prediction = tf.equal(network_pred_class, labels_class)

            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            accuracy_summary = tf.summary.scalar("accuracy", accuracy)

            saver = tf.train.Saver()

            with tf.Session(graph=graph) as session:
                merged_summaries = tf.summary.merge_all()
                now = datetime.datetime.now()
                log_path = "C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/tmp/CIFAR10/log/" + str(now.hour) + str(now.minute) + str(now.second)
                writer_summaries = tf.summary.FileWriter(log_path, graph)

                 # Use TensorFlow to find the latest checkpoint - if any.
                last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)

                self.logMessage.emit('Trying to restore last checkpoint ...')
                try:
                    # Try and load the data in the checkpoint.
                    saver.restore(session, save_path=last_chk_path)
                    self.logMessage.emit('Restored checkpoint from: {} '.format(last_chk_path))
                except:
                    self.logMessage.emit('Failed to restore checkpoint. Initializing variables instead.')
                    # initialise variables if not possible
                    session.run(tf.global_variables_initializer())


                for epoch in range(num_epochs):
                    if self.end_training == False:
                        # grab random batch from training data
                        offset = (epoch * batch_size) % (training_labels.shape[0] - batch_size)
                        batch_data = training_set[offset:(offset + batch_size), :, :, :]
                        batch_labels = training_labels[offset:(offset + batch_size)]

                        # feed batch through network
                        feed_dict = {x: batch_data, y: batch_labels, keep_prob: 0.5}
                        _, l, predictions, my_summary, acc = session.run([optimizer, loss, model_output, merged_summaries, accuracy], 
                                                        feed_dict=feed_dict)

                        writer_summaries.add_summary(my_summary, epoch)

                        # print information to output log
                        if (epoch % 100 == 0):
                            self.logMessage.emit('')
                            self.logMessage.emit('Loss at epoch: {} of {} is {}'.format(epoch, str(num_epochs), l))
                            self.logMessage.emit('Global Step: {}'.format(str(global_step.eval())))
                            self.logMessage.emit('Learning Rate: {}'.format(str(learning_rate)))
                            self.logMessage.emit('Minibatch size: {}'.format(str(batch_labels.shape)))
                            self.logMessage.emit('Batch Accuracy = {}'.format(str(acc)))
                            self.lossOverEpochs.emit(l, epoch)

                        # calculate progress as percentage and emit signal to GUI to update
                        epochProg = (epoch / num_epochs) * 100
                        self.epochProgress.emit(epochProg)

                        # check save interval to avoid divide by zero
                        if save_interval != 0:
                            # save at interval define by user
                            if (epoch % save_interval == 0):
                                saver.save(sess=session, save_path=save_path, global_step=global_step)
                                self.logMessage.emit('Saved Checkpoint \n')

                # set progress bar to 100% (if training was not interrupted)
                if (self.end_training == False):
                    self.epochProgress.emit(100)

                # run test set through network and send value to GUI
                test_acc = session.run(accuracy, feed_dict={x: testing_set, y:testing_labels, keep_prob: 1.0})
                self.testSetAccuracy.emit(test_acc)

                # save network state when complete
                saver.save(sess=session, save_path=save_path, global_step=global_step)
                self.logMessage.emit('\nTraining Complete')
                self.logMessage.emit('Saved Checkpoint \n')

                # signal that thread has completed
                self.workComplete.emit()

    def cancel_training(self):
        self.end_training = True


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100, title='title'):
        self.title = title
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.suptitle(title)

        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.losses = []
        self.epochs = []


class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""
    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)

    def compute_initial_figure(self):
        #self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')
        self.axes.set_ylabel('Loss')
        self.axes.set_xlabel('Epoch')
        self.axes.grid(True)

    @pyqtSlot(float, int)
    def update_figure(self, loss: float, epoch: int):
        self.axes.set_ylabel('Loss')
        self.axes.set_xlabel('Epoch')
        # add current values to array
        self.losses.append(loss)
        self.epochs.append(epoch)
        # plot new, extended graph
        self.axes.plot(self.epochs, self.losses, 'r')
        self.axes.grid(True)
        self.draw()


class CNNApp(QMainWindow, design.Ui_MainWindow):

    # signal to background thread that the user has requested training to cancel
    end_train = pyqtSignal()

    def __init__(self):  
        super().__init__()
        self.setupUi(self)

        # initialise GUI elements
        self.btnCancelTraining.setEnabled(False)
        self.txtConvKeepRate.setEnabled(False)
        self.txtPoolKeepRate.setEnabled(False)
        self.txtFCKeepRate.setEnabled(False)
        self.btnCreateModel.setEnabled(False)

        # initialise threads array, if multiple required
        self.__threads = None
        self.newModel = []

        # default text for fields
        self.txtSavePath.setText('C:/tmp/')
        self.txtLoadCheckpoints.setText('C:/tmp/')
        self.txtModelSavePath.setText('C:/tmp/')
        self.txtLoadModel.setText('C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/Tensorflow-Final-Year-Project/TestModel.xml')

        # navigational buttons
        self.actionTrain.triggered.connect(self.openTrainTab)
        self.actionDesign.triggered.connect(self.openDesignTab)
        self.actionVisualizations.triggered.connect(self.openVisualizationTab)
        self.actionExit.triggered.connect(self.close)


        self.cbxSavePath.stateChanged.connect(self.checkpointCheckBoxStateChange)
        self.radCreateModel.toggled.connect(self.createModelRadClicked)

        # events for checking/unchecking dropout when creating model
        self.cbxConvDropout.stateChanged.connect(self.dropoutCheckBoxStateChanged)
        self.cbxPoolDropout.stateChanged.connect(self.dropoutCheckBoxStateChanged)
        self.cbxFCDropout.stateChanged.connect(self.dropoutCheckBoxStateChanged)
        
        # buttons to change file paths 
        self.btnChangeSavePath.clicked.connect(self.changeCheckpointSavePath)
        self.btnChangeLoadCheckpoints.clicked.connect(self.changeCheckpointLoadPath)
        self.btnChangeModelPath.clicked.connect(self.loadModel)
        self.btnChangeModelSavePath.clicked.connect(self.changeModelSavePath)
        
        # train/cancel training buttons
        self.btnTrainNetwork.clicked.connect(self.trainButtonClicked)
        self.btnCancelTraining.clicked.connect(self.cancel_train)

        # create new model buttons
        self.btnAddConvLayer.clicked.connect(self.createConvLayerButtonClicked)
        self.btnAddMaxPool.clicked.connect(self.createPoolingLayerButtonClicked)
        self.btnAddFullyConn.clicked.connect(self.createDenseLayerButtonClicked)
        self.btnAddOutput.clicked.connect(self.createOutputLayerButtonClicked)
        self.btnValidateNetwork.clicked.connect(self.validateModelButtonClicked)
        self.btnCreateModel.clicked.connect(self.createModelButtonClicked)

        # clear output logs
        self.btnClearLog.clicked.connect(self.clearOutputLog)
        self.btnClearModelLog.clicked.connect(self.clearOutputModelLog)

        self.dc = MyDynamicMplCanvas(self.lossGraphWidget, width=5, height=4, dpi=100, title='Loss Over Epochs')
        #self.graphWidget.addWidget(dc)

        

    def trainButtonClicked(self):
        try:
            '''
            num_epochs = int(self.txtNumEpochs.text())
            batch_size = int(self.txtBatchSize.text())
            learning_rate = float(self.txtLearningRate.text())
            optimizer = int(self.cbxOptimizer.currentIndex())
            '''
            num_epochs = 1000
            batch_size = 64
            learning_rate = 0.005
            optimizer = 0
            
            # obtain model file path and checkpoint save path from user text fields
            model_path = self.txtLoadModel.text()
            save_path = self.txtSavePath.text()

            if self.cbxSaveInterval.currentIndex() == 0:
                save_interval = 0
            else:
                save_interval = int(self.cbxSaveInterval.currentText())
    
        except ValueError:
            self.txtOutputLog.append('Number of Epochs, Batch Size and Learning Rate must be a Numerical Value!')
        else:
            # initialise threads array
            self.__threads = []

            # update GUI
            self.btnTrainNetwork.setDisabled(True)
            self.btnCancelTraining.setEnabled(True)
            self.prgTrainingProgress.setValue(0)

            # check which data set is selected
            if self.radCIFAR10.isChecked():
                data_set = 'CIFAR10'
            elif self.radMNIST.isChecked():
                data_set = 'MNIST'


            worker = Worker(data_set, num_epochs, batch_size, learning_rate, optimizer, save_path, save_interval, model_path)
            thread = QThread()

            # connect cancel button in main thread to background thread
            self.end_train.connect(worker.cancel_training) # THIS WOULD NOT WORK A FEW LINES LOWER!!!!

            # store reference to objects so they are not garbage collected
            self.__threads.append((thread, worker))
            worker.moveToThread(thread)

            # set connections from background thread to main thread for updating GUI
            worker.testSetAccuracy.connect(self.updateTestSetAccuracy)
            worker.epochProgress.connect(self.updateProgressBar)
            worker.lossOverEpochs.connect(self.dc.update_figure)
            worker.logMessage.connect(self.txtOutputLog.append)
            worker.workComplete.connect(self.abort_workers)

            # connect and start thread
            thread.started.connect(worker.work)
            thread.start()

    def createConvLayerButtonClicked(self):

        if not self.newModel:
            if self.currentDataSet() == 'CIFAR10':
                num_inputs = 3
            elif self.currentDataSet() == 'MNIST':
                num_inputs = 1
        else:
            for e in self.newModel[::-1]:
                if e.layerType == 'Convolution':
                    num_inputs = e.numOutputFilters
                    break

        try:
            conv_kernel_size = int(self.txtConvKernelSize.text())
            conv_stride = int(self.txtConvStride.text())
            num_output_filters = int(self.txtConvOutputFilters.text())

            if self.cbxConvDropout.isChecked():
                keep_rate = float(self.txtConvKeepRate.text())
            else: keep_rate = 1.0

            if self.cbxConvActFunction.currentIndex() == 0:
                act_function = 'ReLu'
            elif self.cbxConvActFunction.currentIndex() == 1:
                act_function = 'Sigmoid'

            if self.cbxConvPadding.currentIndex() == 0:
                padding = 'SAME'

            if self.cbxConvNorm.isChecked():
                normalize = True
            else:
                normalize = False

            if self.cbxConvDropout.isChecked():
                dropout = True
            else:
                dropout = False

            layer = l.ConvLayer('Convolution', num_inputs, conv_kernel_size, conv_stride, act_function, num_output_filters, padding, normalize, dropout, keep_rate)

            self.newModel.append(layer)

            item = QListWidgetItem(("Convolution, Num Inputs: {}, Num Output Filters {}, Kernel Size: {}, Stride: {}, Activation Function: {}, Padding: {}, Normalize: {}, Dropout: {}, Keep Rate: {}").format(layer.numInputs, layer.numOutputFilters, layer.kernelSize, layer.stride, layer.actFunction, layer.padding, layer.normalize, layer.dropout, layer.keepRate))
            
            self.lstModel.addItem(item)

        except ValueError:
            self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

    def createPoolingLayerButtonClicked(self):
        try:    
            pool_kernel_size = int(self.txtPoolKernelSize.text())
            pool_stride = int(self.txtPoolStride.text())

            if self.cbxConvDropout.isChecked():
                keep_rate = float(self.txtConvKeepRate.text())
            else: keep_rate = 1.0

            if self.cbxPoolPadding.currentIndex() == 0:
                padding = 'SAME'

            if self.cbxPoolNorm.isChecked():
                normalize = True
            else:
                normalize = False

            if self.cbxPoolDropout.isChecked():
                dropout = True
            else:
                dropout = False

            layer = l.MaxPoolingLayer('Max Pool',  pool_kernel_size, pool_stride, padding, normalize, dropout, keep_rate)
            self.newModel.append(layer)

            item = QListWidgetItem(("Max Pool, Kernel Size: {}, Stride: {}, Padding: {}, Normalize: {}, Dropout: {}, Keep Rate: {}").format(layer.kernelSize, layer.stride, layer.padding, layer.normalize, layer.dropout, layer.keepRate))
            self.lstModel.addItem(item)

        except ValueError:
            self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

    def createDenseLayerButtonClicked(self):

        num_prev_dense_layers = 0
        for e in range(len(self.newModel)):
            if self.newModel[e].layerType == 'Dense':
                num_prev_dense_layers += 1

        if num_prev_dense_layers == 0:
            if self.currentDataSet() == 'CIFAR10':
                image_size = 32
            if self.currentDataSet() == 'MNIST':
                image_size = 28

            for e in self.newModel:
                if e.layerType == 'Max Pool':
                    image_size /= e.stride

            for e in self.newModel[::-1]:
                if e.layerType == 'Convolution':
                    prev_filter_size = e.numOutputFilters
                    break
            num_inputs = image_size * image_size * prev_filter_size

        else:
            num_inputs = self.newModel[-1].numOutputNodes

        try:
            if self.cbxFCActFunction.currentIndex() == 0:
                act_function = 'ReLu'
            elif self.cbxFCActFunction.currentIndex() == 1:
                act_function = 'Sigmoid'

            num_output_nodes = int(self.txtFCNumOutputNodes.text())

            if self.cbxFCDropout.isChecked():
                keep_rate = float(self.txtFCKeepRate.text())
            else: keep_rate = 1.0

            if self.cbxFCNorm.isChecked():
                normalize = True
            else:
                normalize = False

            if self.cbxFCDropout.isChecked():
                dropout = True
            else:
                dropout = False

            layer = l.DenseLayer('Dense',  num_inputs, act_function, num_output_nodes, normalize, dropout, keep_rate)
            self.newModel.append(layer)

            item = QListWidgetItem(("Dense, Num Inputs: {}, Num Output Nodes: {}, Activation Function: {}, Normalize: {}, Dropout: {}, Keep Rate: {}").format(layer.numInputs, layer.numOutputNodes, layer.actFunction, layer.normalize, layer.dropout, layer.keepRate))
            self.lstModel.addItem(item)

        except ValueError:
            self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

    def createOutputLayerButtonClicked(self):

        num_inputs = self.newModel[-1].numOutputNodes

        if self.currentDataSet() == 'CIFAR10' or self.currentDataSet() == 'MNIST':
            num_outputs = 10

        try:
            if self.cbxOutputActFunction.currentIndex() == 0:
                act_function = 'ReLu'
            elif self.cbxOutputActFunction.currentIndex() == 1:
                act_function = 'Sigmoid'

            layer = l.OutputLayer('Output', num_inputs, act_function, num_outputs)
            self.newModel.append(layer)

            item = QListWidgetItem(("Output, Num Inputs: {}, Num Output Nodes: {}, Activation Function: {}").format(layer.numInputs, layer.numOutputNodes, layer.actFunction))
            self.lstModel.addItem(item)

        except ValueError:
            self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

    def validateModelButtonClicked(self):
        self.btnCreateModel.setEnabled(True)

    def createModelButtonClicked(self):
        success = pp.createXMLModel(self.newModel)
        if success == True:
            self.txtOutputModelLog.append("Success Writing XML File")
        else : 
            self.txtOutputModelLog.append("Error Writing XML File")


    def changeCheckpointSavePath(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.txtSavePath.setText(path + "/")

    def changeCheckpointLoadPath(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.txtLoadCheckpoints.setText(path + "/")
        self.cbxSavePath.setCheckState(False)

    def changeModelSavePath(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.txtModelSavePath.setText(path + "/")

    def currentDataSet(self):
        if self.radCIFAR10Model.isChecked():
            return 'CIFAR10'
        elif self.radMNISTModel.isChecked():
            return 'MNIST'

    def loadModel(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home', "XML Files (*.xml)")
        self.txtLoadModel.setText(str(fname[0])) 

    def checkpointCheckBoxStateChange(self):
        if self.cbxSavePath.isChecked():
            self.txtSavePath.setText(self.txtLoadCheckpoints.text())

    def dropoutCheckBoxStateChanged(self):
        if self.cbxConvDropout.isChecked():
            self.txtConvKeepRate.setEnabled(True)
        else: self.txtConvKeepRate.setEnabled(False)

        if self.cbxPoolDropout.isChecked():
            self.txtPoolKeepRate.setEnabled(True)
        else: self.txtPoolKeepRate.setEnabled(False)

        if self.cbxFCDropout.isChecked():
            self.txtFCKeepRate.setEnabled(True)
        else: self.txtFCKeepRate.setEnabled(False)

    def createModelRadClicked(self, enabled):
        if enabled:
            self.tabWidget.setCurrentIndex(1)
            self.radLoadModel.setChecked(True)
            
    def openTrainTab(self):
        self.tabWidget.setCurrentIndex(0)

    def openDesignTab(self):
        self.tabWidget.setCurrentIndex(1)

    def openVisualizationTab(self):
        self.tabWidget.setCurrentIndex(2)

    def clearOutputLog(self):
        self.txtOutputLog.setText('')

    def clearOutputModelLog(self):
        self.txtOutputModelLog.setText('')

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "Are you sure you want to quit? All Unsaved Progress will be lost...", 
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
    

    @pyqtSlot(float)
    def updateTestSetAccuracy(self, accuracy: float):
        self.txtTestAccuracy.setText(str(round(accuracy, 2)))
        self.btnTrainNetwork.setEnabled(True)

    @pyqtSlot(float)
    def updateProgressBar(self, progress: float):
        self.prgTrainingProgress.setValue(progress)

    # called when thread(s) have finished, i.e. training has finished or been cancelled
    @pyqtSlot()
    def abort_workers(self):
        for thread, worker in self.__threads:  # note nice unpacking by Python, avoids indexing
            thread.quit()  # this will quit **as soon as thread event loop unblocks**
            thread.wait()  # <- so you need to wait for it to *actually* quit
    
    def cancel_train(self):
        self.end_train.emit()
        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    ui = CNNApp()
    ui.show()

    sys.exit(app.exec_())
