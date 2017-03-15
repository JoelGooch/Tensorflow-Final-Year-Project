import sys
import tensorflow as tf
import numpy as np
import os
import pickle
import datetime
import cv2
import matplotlib
import CNN_GUI_V8 as design
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import practiceParse as pp
from tensorflow.examples.tutorials.mnist import input_data

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
def layer_factory(inputLayer, layer):
    if layer.layerType == 'Convolution':
        newLayer = new_conv_layer(inputLayer, layer.numInputs, layer.kernelSize, layer.stride, layer.actFunction, 
            layer.numOutputFilters, layer.padding, layer.normalize, layer.dropout, layer.keepRate)
    elif layer.layerType == 'Max Pool':
        newLayer = new_max_pool_layer(inputLayer, layer.kernelSize, layer.stride, layer.padding, layer.normalize, layer.dropout, layer.keepRate)
    elif layer.layerType == 'Dense':
        newLayer = new_dense_layer(inputLayer, layer.numInputs, layer.numOutputNodes, layer.actFunction, layer.normalize, layer.dropout, layer.keepRate)
    elif layer.layerType == 'Output':
        newLayer = new_output_layer(inputLayer, layer.numInputs, layer.numOutputNodes, layer.actFunction)
    return newLayer

# function to load data set and parameters for CIFAR10
def load_CIFAR_10():
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

            print(images_batch.shape)

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

    return training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes

# function to loead data set and parameters for MNIST
def load_MNIST():
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

    return training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes


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
            training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes = load_CIFAR_10()
        if (data_set == 'MNIST'):
            training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes = load_MNIST()

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

                # create as many layers as are stored in XML file
                if numLayers > 0:
                    layer1 = layer_factory(x, Layers[0])
                    networkLayers.append(layer1)
                if numLayers > 1:
                    layer2 = layer_factory(layer1, Layers[1])
                    networkLayers.append(layer2)
                if numLayers > 2:
                    layer3 = layer_factory(layer2, Layers[2])
                    networkLayers.append(layer3)
                if numLayers > 3:
                    layer4 = layer_factory(layer3, Layers[3])
                    layer4 = flatten_layer(layer4)                   # NEED TO WORK OUT HOW TO NOT HARD CODE THIS HERE
                    networkLayers.append(layer4)
                if numLayers > 4:
                    layer5 = layer_factory(layer4, Layers[4])
                    networkLayers.append(layer5)
                if numLayers > 5:
                    layer6 = layer_factory(layer5, Layers[5])
                    networkLayers.append(layer6)
                if numLayers > 6:
                    layer7 = layer_factory(layer6, Layers[6])
                    networkLayers.append(layer7)
                if numLayers > 7:
                    layer8 = layer_factory(layer7, Layers[7])
                    networkLayers.append(layer8)
                if numLayers > 8:
                    layer9 = layer_factory(layer8, Layers[8])
                    networkLayers.append(layer9)

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
                optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
                self.logMessage.emit('Optimizer: Adam')
            elif (learning_algo == 2):
                optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
                self.logMessage.emit('Optimizer: Ada Grad')
            #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95).minimize(loss, global_step=global_step)


            network_pred_class = tf.argmax(model_output, dimension=1)
            correct_prediction = tf.equal(network_pred_class, labels_class)

            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            accuracy_summary = tf.summary.scalar("accuracy", accuracy)

            saver = tf.train.Saver()

            with tf.Session(graph=graph) as session:
                merged_summaries = tf.summary.merge_all()
                now = datetime.datetime.now()
                log_path = "C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/tmp/CIFAR10/log/" + str(now.hour) + str(now.minute) + str(now.second)
                #save_dir = os.path.join(save_dir, 'CIFAR10')
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
                    session.run(tf.global_variables_initializer())


                for epoch in range(num_epochs):
                    if self.end_training == False:
                        offset = (epoch * batch_size) % (training_labels.shape[0] - batch_size)
                        batch_data = training_set[offset:(offset + batch_size), :, :, :]
                        batch_labels = training_labels[offset:(offset + batch_size)]

                        feed_dict = {x: batch_data, y: batch_labels, keep_prob: 0.5}
                        _, l, predictions, my_summary, acc = session.run([optimizer, loss, model_output, merged_summaries, accuracy], 
                                                        feed_dict=feed_dict)
                        writer_summaries.add_summary(my_summary, epoch)

                        if (epoch % 100 == 0):
                            self.logMessage.emit('')
                            self.logMessage.emit('Loss at epoch: {} of {} is {}'.format(epoch, str(num_epochs), l))
                            self.logMessage.emit('Global Step: {}'.format(str(global_step.eval())))
                            self.logMessage.emit('Learning Rate: {}'.format(str(learning_rate)))
                            self.logMessage.emit('Minibatch size: {}'.format(str(batch_labels.shape)))
                            self.logMessage.emit('Batch Accuracy = {}'.format(str(acc)))
                            self.lossOverEpochs.emit(l, epoch)

                        epochProg = (epoch / num_epochs) * 100
                        self.epochProgress.emit(epochProg)

                        if save_interval != 0:
                            if (epoch % save_interval == 0):
                                saver.save(sess=session, save_path=save_path)
                                print("Saved Checkpoint")

                if (self.end_training == False):
                    self.epochProgress.emit(100)

                test_acc = session.run(accuracy, feed_dict={x: testing_set, y:testing_labels, keep_prob: 1.0})
                self.testSetAccuracy.emit(test_acc)
                saver.save(session, save_path=save_path, global_step=global_step)
                self.logMessage.emit('\nTraining Complete')
                self.logMessage.emit('Saved Checkpoint \n')
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

        # initialise threads array, if multiple required
        self.__threads = None

        # default text for fields
        self.txtSavePath.setText('C:/tmp/')
        self.txtLoadCheckpoints.setText('C:/tmp/')
        self.txtLoadModel.setText('C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/Tensorflow-Final-Year-Project/MNISTModel.xml')

        # navigational buttons
        self.actionTrain.triggered.connect(self.openTrainTab)
        self.actionDesign.triggered.connect(self.openDesignTab)
        self.actionVisualizations.triggered.connect(self.openVisualizationTab)
        self.actionExit.triggered.connect(self.close)


        self.chkSavePath.stateChanged.connect(self.checkpointCheckBoxStateChange)
        self.radCreateModel.toggled.connect(self.createModelRadClicked)
        
        # buttons to change file paths 
        self.btnChangeSavePath.clicked.connect(self.changeCheckpointSavePath)
        self.btnChangeLoadCheckpoints.clicked.connect(self.changeCheckpointLoadPath)
        self.btnChangeModelPath.clicked.connect(self.loadModel)
        
        # train/cancel training buttons
        self.btnTrainNetwork.clicked.connect(self.trainButtonClicked)

        self.btnCancelTraining.clicked.connect(self.cancel_train)

        self.btnCancelTraining.setDisabled(True)


        self.dc = MyDynamicMplCanvas(self.lossGraphWidget, width=5, height=4, dpi=100, title='Loss Over Epochs')
        #self.graphWidget.addWidget(dc)

        

    def trainButtonClicked(self):
        try:
            '''
            num_epochs = int(self.txtNumEpochs.text())
            batch_size = int(self.txtBatchSize. text())
            learning_rate = float(self.txtLearningRate.text())
            optimizer = int(self.cbxOptimizer.currentIndex())
            '''
            num_epochs = 1000
            batch_size = 64
            learning_rate = 0.05
            optimizer = 1
            
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

        
            thread.started.connect(worker.work)
            thread.start()

    def changeCheckpointSavePath(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.txtSavePath.setText(path + "/")

    def changeCheckpointLoadPath(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.txtLoadCheckpoints.setText(path + "/")
        self.chkSavePath.setCheckState(False)

    def loadModel(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home', "XML Files (*.xml)")
        self.txtLoadModel.setText(str(fname[0])) 

    def checkpointCheckBoxStateChange(self):
        if self.chkSavePath.isChecked():
            self.txtSavePath.setText(self.txtLoadCheckpoints.text())

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
