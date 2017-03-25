import sys
import tensorflow as tf
import numpy as np
import pickle
import datetime
import cv2
import matplotlib as mpl
mpl.use('Agg')
import xml.etree.ElementTree as ET

import CNN_GUI_V12 as design
import input_data as data
import xml_parser as pp
import Layer as l
import layer_factory as factory

from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog, QSizePolicy, QListWidgetItem
from PyQt5.QtGui import QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

#np.set_printoptions(threshold=np.inf)


class Worker(QObject):

	epochProgress = pyqtSignal(float)
	testSetAccuracy = pyqtSignal(float)
	logMessage = pyqtSignal(str)
	workComplete = pyqtSignal()

	lossOverEpochs = pyqtSignal(float, int)
	testSetAccOverEpochs = pyqtSignal(float, int)
	confusionMat = pyqtSignal(object, bool, int)

	networkModel = pyqtSignal(list)

	def __init__(self, data_set: str, num_epochs: int, batch_size: int, learning_rate: float, optimizer: int, l2_reg: bool, beta: float, save_directory: str, save_interval: int, model_path: str, run_time: bool):
		super().__init__()
		self.end_training = False
		self.data_set = data_set
		self.num_epochs = num_epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.optimizer = optimizer
		self.l2_reg = l2_reg
		self.beta = beta
		self.save_directory = save_directory
		self.save_interval = save_interval
		self.model_path = model_path
		self.run_time = run_time
		
	@pyqtSlot()
	def work(self):
		self.train_network(self.data_set, self.num_epochs, self.batch_size, self.learning_rate, self.optimizer, self.l2_reg, self.beta, self.save_directory, self.save_interval, self.model_path, self.run_time)


	def train_network(self, data_set, num_epochs, batch_size, learning_rate, learning_algo, l2_reg, beta, save_path, save_interval, model_path, run_time):
		
		run_time = False

		# load selected data set
		if (data_set == 'CIFAR10'):
			training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_CIFAR_10()
		if (data_set == 'MNIST'):
			training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_MNIST()
		if (data_set == 'Prima head pose'):
			training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_prima_head_pose()

		# normalize data, THIS BREAKS MNIST, CIFAR10 WORKS WITH OR WITHOUT
		#training_set -= 127 
		#testing_set -= 127

		try:
			# get array of layers from XML file
			Layers = pp.getLayers(model_path)
		except:
			self.logMessage.emit('Error reading XML file')
			return 0

		self.networkModel.emit(Layers)

		self.logMessage.emit('Initialising Tensorflow Variables... \n')
		
		graph = tf.Graph()
		with graph.as_default():

			# define placeholder variables
			x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
			y = tf.placeholder(tf.float32, shape=(None, num_classes))
			keep_prob = tf.placeholder(tf.float32)

			# stores the class integer values 
			labels_class = tf.argmax(y, dimension=1)
			
			numLayers = len(Layers)

			# array to hold tensorflow layers
			networkLayers = []

			def CNN_Model(data, _dropout=1.0):

				# instantiate layer factory
				layerFactory = factory.layer_factory(data_set, Layers)

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

			if data_set == 'MNIST' or data_set == 'CIFAR10':
				loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
			elif data_set == 'Prima head pose':
				loss = tf.nn.l2_loss(model_output - y)

			# add l2 regularization if user has specified
			if l2_reg == True:
				# for all weights of network, add regularization term
				for e in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
					loss += (beta * tf.nn.l2_loss(e))
			

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
			elif (learning_algo == 3):
				optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step)
				self.logMessage.emit('Optimizer: Ada Delta')
			#elif (learning_algo == 4):
			#	optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=global_step)
			#	self.logMessage.emit('Optimizer: Momentum')


			# convert one hot encoded array to single prediction value
			network_pred_class = tf.argmax(model_output, dimension=1)

			# compare if network prediction was correct
			correct_prediction = tf.equal(network_pred_class, labels_class)
			
			# define network accuracy
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			# create saver object to save/restore checkpoints
			saver = tf.train.Saver()

			# begin session
			with tf.Session(graph=graph) as session:

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


				# start training loop
				for epoch in range(num_epochs + 1):
					# check if user has signalled training to be cancelled
					if self.end_training == False:
						# grab random batch from training data
						offset = (epoch * batch_size) % (training_labels.shape[0] - batch_size)
						batch_data = training_set[offset:(offset + batch_size), :, :, :]
						batch_labels = training_labels[offset:(offset + batch_size)]

						# feed batch through network
						feed_dict = {x: batch_data, y: batch_labels, keep_prob: 0.5}
						_, l, predictions, acc, batch_pred_class, batch_classes = session.run([optimizer, loss, model_output, accuracy, network_pred_class, labels_class], feed_dict=feed_dict)

						# print information to output log
						if (epoch % 100 == 0):
							self.logMessage.emit('')
							self.logMessage.emit('Loss at epoch: {} of {} is {}'.format(epoch, str(num_epochs), l))
							self.logMessage.emit('Global Step: {}'.format(str(global_step.eval())))
							self.logMessage.emit('Learning Rate: {}'.format(str(learning_rate)))
							self.logMessage.emit('Minibatch size: {}'.format(str(batch_labels.shape)))
							self.logMessage.emit('Batch Accuracy = {}'.format(str(acc)))

							# emit batch loss for GUI visualization
							self.lossOverEpochs.emit(l, epoch)

							# if user has chosen to include run time visualizations
							if run_time == True:
								# create confusion matrix for current batch
								batch_confusion = tf.contrib.metrics.confusion_matrix(batch_classes, batch_pred_class).eval()
								self.confusionMat.emit(batch_confusion, True, epoch)

								# run test set at current batch
								test_accuracy_epoch  = session.run(accuracy, feed_dict={x: testing_set, y:testing_labels, keep_prob: 1.0})
								self.testSetAccOverEpochs.emit(test_accuracy_epoch, epoch)

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

				# run test set through network and return accuracy and predicted classes
				test_acc, testing_pred_class, testing_classes = session.run([accuracy, network_pred_class, labels_class], 
												feed_dict={x: testing_set, y:testing_labels, keep_prob: 1.0})
				
				# send test accuracy to GUI
				self.testSetAccuracy.emit(test_acc)

				# create confusion matrix from predicted and actual classes
				confusion = tf.contrib.metrics.confusion_matrix(testing_classes, testing_pred_class).eval()
				self.confusionMat.emit(confusion, False, 0)

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
	def __init__(self, parent=None, width=5, height=4, dpi=100, title='title', xAxisTitle='x', yAxisTitle='y'):
		self.title = title

		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = self.fig.add_subplot(111)
		self.fig.suptitle(title)

		self.axes.set_xlabel(xAxisTitle)
		self.axes.set_ylabel(yAxisTitle)
		self.axes.grid(True)

		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)

		FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)


class ErrorOverEpochsGraph(MyMplCanvas):
	"""A canvas that updates itself every secoSnd with a new plot."""
	def __init__(self, *args, **kwargs):
		MyMplCanvas.__init__(self, *args, **kwargs)
		self.errors = []
		self.epochs = []

	@pyqtSlot(float, int)
	def update_figure(self, error: float, epoch: int):

		if epoch == 0:
			self.axes.clear()
			self.errors = []
			self.epochs = []

		self.axes.set_ylabel('Error')
		self.axes.set_xlabel('Epoch')
		# add current values to array
		self.errors.append(error)
		self.epochs.append(epoch)
		# plot new, extended graph
		self.axes.plot(self.epochs, self.errors, 'r')
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
		self.txtL2Beta.setEnabled(False)

		# initialise threads array, if multiple required
		self.__threads = None
		self.newModel = []

		# default text for fields
		self.txtSavePath.setText('C:/tmp/')
		self.txtLoadCheckpoints.setText('C:/tmp/')
		self.txtModelSavePath.setText('C:/tmp/')
		self.txtLoadModel.setText('C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/Tensorflow-Final-Year-Project/Models/MNISTmodel.xml')

		# navigational buttons
		self.actionTrain.triggered.connect(self.openTrainTab)
		self.actionDesign.triggered.connect(self.openDesignTab)
		self.actionVisualizations.triggered.connect(self.openVisualizationTab)
		self.actionExit.triggered.connect(self.close)

		self.cbxSavePath.stateChanged.connect(self.checkpointCheckBoxStateChange)
		self.radCreateModel.toggled.connect(self.createModelRadClicked)

		# events for checking/unchecking dropout when creating model
		self.chkConvDropout.stateChanged.connect(self.dropoutCheckBoxStateChanged)
		self.chkPoolDropout.stateChanged.connect(self.dropoutCheckBoxStateChanged)
		self.chkFCDropout.stateChanged.connect(self.dropoutCheckBoxStateChanged)


		self.chkL2Reg.stateChanged.connect(self.l2RegularizationCheckBoxStateChanged)

		self.cbxConvBiasInit.currentIndexChanged.connect(self.biasInitComboBoxChanged)
		
		# buttons to change file paths 
		self.btnChangeSavePath.clicked.connect(self.changeCheckpointSavePath)
		self.btnChangeLoadCheckpoints.clicked.connect(self.changeCheckpointLoadPath)
		self.btnChangeModelPath.clicked.connect(self.changeModelLoadPath)
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
		self.btnDeleteModel.clicked.connect(self.deleteModelButtonClicked)
		self.btnDeleteLayer.clicked.connect(self.deleteLastLayerButtonClicked)

		# clear output logs
		self.btnClearLog.clicked.connect(self.clearOutputLog)
		self.btnClearModelLog.clicked.connect(self.clearOutputModelLog)

		# create graph instances
		self.lossGraph = ErrorOverEpochsGraph(self.lossGraphWidget, width=5, height=4, dpi=100, title='Loss Over Epochs', xAxisTitle='Epoch', yAxisTitle='Loss')
		self.testErrorGraph = ErrorOverEpochsGraph(self.graphWidget2, width=5, height=4, dpi=100, title='Test Error Over Epochs', xAxisTitle='Epoch', yAxisTitle='Error')

		
	def trainButtonClicked(self):
		try:
			'''
			num_epochs = int(self.txtNumEpochs.text())
			batch_size = int(self.txtBatchSize.text())
			learning_rate = float(self.txtLearningRate.text())
			optimizer = int(self.cbxOptimizer.currentIndex())
			'''
			num_epochs = 200
			batch_size = 128
			learning_rate = 0.005
			optimizer = 0
			
			# obtain model file path and checkpoint save path from user text fields
			model_path = self.txtLoadModel.text()
			save_path = self.txtSavePath.text()

			if self.chkL2Reg.isChecked():
				l2_reg = True
				beta = float(self.txtL2Beta.text())
			else:
				l2_reg = False
				beta = 0

			if self.chkVisualizations.isChecked():
				run_time = True
			else:
				run_time = False

			if self.cbxSaveInterval.currentIndex() == 0:
				save_interval = 0
			else:
				save_interval = int(self.cbxSaveInterval.currentText())
	
		except ValueError:
			self.txtOutputLog.append('Number of Epochs, Batch Size and Learning Rate and L2 Beta must be a Numerical Value!')
		else:
			# initialise threads array
			self.__threads = []

			# update GUI
			self.btnTrainNetwork.setDisabled(True)
			self.btnCancelTraining.setEnabled(True)
			self.prgTrainingProgress.setValue(0)

			worker = Worker(self.currentDataSet(), num_epochs, batch_size, learning_rate, optimizer, l2_reg, beta, save_path, save_interval, model_path, run_time)
			thread = QThread()

			# connect cancel button in main thread to background thread
			self.end_train.connect(worker.cancel_training) # THIS WOULD NOT WORK A FEW LINES LOWER!!!!

			# store reference to objects so they are not garbage collected
			self.__threads.append((thread, worker))
			worker.moveToThread(thread)

			# set connections from background thread to main thread for updating GUI
			worker.testSetAccuracy.connect(self.updateTestSetAccuracy)
			worker.epochProgress.connect(self.updateProgressBar)

			worker.lossOverEpochs.connect(self.lossGraph.update_figure)
			worker.testSetAccOverEpochs.connect(self.testErrorGraph.update_figure)
			worker.confusionMat.connect(self.updateConfusionPlot)

			worker.logMessage.connect(self.txtOutputLog.append)
			worker.networkModel.connect(self.showModelDetails)
			worker.workComplete.connect(self.abort_workers)

			# connect and start thread
			thread.started.connect(worker.work)
			thread.start()

	def createConvLayerButtonClicked(self):

		try:
			conv_layer_name = self.txtConvName.text()
			conv_kernel_size = int(self.txtConvKernelSize.text())
			conv_stride = int(self.txtConvStride.text())
			num_output_filters = int(self.txtConvOutputFilters.text())

			if self.cbxConvActFunction.currentIndex() == 0:
				act_function = 'ReLu'
			elif self.cbxConvActFunction.currentIndex() == 1:
				act_function = 'Sigmoid'
			elif self.cbxConvActFunction.currentIndex() == 2:
				act_function = 'Tanh'

			if self.cbxConvWeightInit.currentIndex() == 0:
				weightInit = "Random Normal"
				weightStdDev = float(self.txtConvStdDev.text())
			elif self.cbxConvWeightInit.currentIndex() == 1:
				weightInit = "Truncated Normal"
				weightStdDev = float(self.txtConvStdDev.text())

			if self.cbxConvBiasInit.currentIndex() == 0:
				biasInit = "Random Normal"
				biasVal = float(self.txtConvBiasVal.text())
			elif self.cbxConvBiasInit.currentIndex() == 1:
				biasInit = "Truncated Normal"
				biasVal = float(self.txtConvBiasVal.text())
			elif self.cbxConvBiasInit.currentIndex() == 2:
				biasInit = "Zeros"
				biasVal = 0
			elif self.cbxConvBiasInit.currentIndex() == 3:
				biasInit = "Constant"
				biasVal = float(self.txtConvBiasVal.text())


			if self.cbxConvPadding.currentIndex() == 0:
				padding = 'SAME'
			elif self.cbxConvPadding.currentIndex() == 1:
				padding = 'VALID'

			if self.chkConvNorm.isChecked():
				normalize = True
			else:
				normalize = False

			if self.chkConvDropout.isChecked():
				dropout = True
				keep_rate = float(self.txtConvKeepRate.text())
			else: 
				dropout = False
				keep_rate = 1.0

			layer = l.ConvLayer(conv_layer_name, conv_kernel_size, conv_stride, act_function, num_output_filters, weightInit, weightStdDev, biasInit, biasVal, padding, normalize, dropout, keep_rate)

			self.newModel.append(layer)

			item = QListWidgetItem(("Convolution, Num Output Filters {}, Kernel Size: {}, Stride: {}, Activation Function: {}, Padding: {}, Normalize: {}, Dropout: {}, Keep Rate: {}").format(layer.numOutputFilters, layer.kernelSize, layer.stride, layer.actFunction, layer.padding, layer.normalize, layer.dropout, layer.keepRate))
			
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	def createPoolingLayerButtonClicked(self):
		try:    
			pool_layer_name = self.txtPoolName.text()
			pool_kernel_size = int(self.txtPoolKernelSize.text())
			pool_stride = int(self.txtPoolStride.text())

			if self.cbxPoolPadding.currentIndex() == 0:
				padding = 'SAME'
			elif self.cbxPoolPadding.currentIndex() == 1:
				padding = 'VALID'

			if self.chkPoolNorm.isChecked():
				normalize = True
			else:
				normalize = False

			if self.chkConvDropout.isChecked():
				dropout = True
				keep_rate = float(self.txtConvKeepRate.text())
			else: 
				dropout = False
				keep_rate = 1.0

			layer = l.MaxPoolingLayer(pool_layer_name, pool_kernel_size, pool_stride, padding, normalize, dropout, keep_rate)
			self.newModel.append(layer)

			item = QListWidgetItem(("Max Pool, Kernel Size: {}, Stride: {}, Padding: {}, Normalize: {}, Dropout: {}, Keep Rate: {}").format(layer.kernelSize, layer.stride, layer.padding, layer.normalize, layer.dropout, layer.keepRate))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	def createDenseLayerButtonClicked(self):

		try:
			dense_layer_name = self.txtFCName.text()

			if self.cbxFCActFunction.currentIndex() == 0:
				act_function = 'ReLu'
			elif self.cbxFCActFunction.currentIndex() == 1:
				act_function = 'Sigmoid'
			elif self.cbxFCActFunction.currentIndex() == 2:
				act_function = 'Tanh'
			num_output_nodes = int(self.txtFCNumOutputNodes.text())

			if self.cbxFCWeightInit.currentIndex() == 0:
				weightInit = "Random Normal"
				weightStdDev = float(self.txtFCStdDev.text())
			elif self.cbxFCWeightInit.currentIndex() == 1:
				weightInit = "Truncated Normal"
				weightStdDev = float(self.txtFCStdDev.text())

			if self.cbxFCBiasInit.currentIndex() == 0:
				biasInit = "Random Normal"
				biasVal = float(self.txtFCBiasVal.text())
			elif self.cbxConvBiasInit.currentIndex() == 1:
				biasInit = "Truncated Normal"
				biasVal = float(self.txtFCBiasVal.text())
			elif self.cbxConvBiasInit.currentIndex() == 2:
				biasInit = "Zeros"
				biasVal = 0
			elif self.cbxFCBiasInit.currentIndex() == 3:
				biasInit = "Constant"
				biasVal = float(self.txtFCBiasVal.text())

			if self.chkFCDropout.isChecked():
				dropout = True
				keep_rate = float(self.txtFCKeepRate.text())
			else: 
				dropout = False
				keep_rate = 1.0

			if self.chkFCNorm.isChecked():
				normalize = True
			else:
				normalize = False			

			layer = l.DenseLayer(dense_layer_name, act_function, num_output_nodes, weightInit, weightStdDev, biasInit, biasVal,  normalize, dropout, keep_rate)
			self.newModel.append(layer)

			item = QListWidgetItem(("Dense,  Num Output Nodes: {}, Activation Function: {}, Normalize: {}, Dropout: {}, Keep Rate: {}").format(layer.numOutputNodes, layer.actFunction, layer.normalize, layer.dropout, layer.keepRate))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	def createOutputLayerButtonClicked(self):
		try:
			output_layer_name = self.txtOutputName.text()

			if self.cbxOutputActFunction.currentIndex() == 0:
				act_function = 'None'
			elif self.cbxOutputActFunction.currentIndex() == 1:
				act_function = 'ReLu'
			elif self.cbxOutputActFunction.currentIndex() == 2:
				act_function = 'Sigmoid'
			elif self.cbxOutputActFunction.currentIndex() == 3:
				act_function = 'Tanh'

			if self.cbxOutputWeightInit.currentIndex() == 0:
				weightInit = "Random Normal"
				weightStdDev = float(self.txtOutputStdDev.text())
			elif self.cbxOutputWeightInit.currentIndex() == 1:
				weightInit = "Truncated Normal"
				weightStdDev = float(self.txtOutputStdDev.text())

			if self.cbxOutputBiasInit.currentIndex() == 0:
				biasInit = "Random Normal"
				biasVal = float(self.txtOutputBiasVal.text())
			elif self.cbxOutputBiasInit.currentIndex() == 1:
				biasInit = "Truncated Normal"
				biasVal = float(self.txtOutputBiasVal.text())
			elif self.cbxOutputBiasInit.currentIndex() == 2:
				biasInit = "Zeros"
				biasVal = 0
			elif self.cbxOutputBiasInit.currentIndex() == 3:
				biasInit = "Constant"
				biasVal = float(self.txtOutputBiasVal.text())

			layer = l.OutputLayer(output_layer_name, act_function, weightInit, weightStdDev, biasInit, biasVal)
			self.newModel.append(layer)

			item = QListWidgetItem(("Output, Activation Function: {}").format(layer.actFunction))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	def validateModelButtonClicked(self):

		if not self.newModel:
			self.txtOutputModelLog.append('No Layers Added')
			return

		if not self.newModel[0].layerType == 'Convolution':
			self.txtOutputModelLog.append('First layer must be Convolution layer')
			return

		if not self.newModel[-1].layerType == 'Output':
			self.txtOutputModelLog.append('Final layer must be output layer')
			return

		self.txtOutputModelLog.append('Model Successfully Validated')
		self.btnCreateModel.setEnabled(True)

	def createModelButtonClicked(self):

		try:
			fileName = self.txtSaveModelAs.text()
			if not fileName:
				self.txtOutputModelLog.append("Please add name for new model")
				return False

			filePath = self.txtModelSavePath.text()

			success = pp.createXMLModel(self.newModel, fileName, filePath)

			if success == True:
				self.txtOutputModelLog.append("Success Writing XML File")
				self.resetModelCreation()
			else : 
				self.txtOutputModelLog.append("Error Writing XML File")
				return False
		except:
			self.txtOutputModelLog.append("Error Writing XML File")

			
		
	def resetModelCreation(self):
		self.deleteModelButtonClicked()
		self.txtConvName.setText('')
		self.txtConvKernelSize.setText('')
		self.txtConvStride.setText('')
		self.cbxConvActFunction.setCurrentIndex(0)
		self.txtConvOutputFilters.setText('')
		self.cbxConvWeightInit.setCurrentIndex(0)
		self.cbxConvBiasInit.setCurrentIndex(0)
		self.txtConvStdDev.setText('')
		self.txtConvBiasVal.setText('')
		self.cbxConvPadding.setCurrentIndex(0)
		self.chkConvNorm.setChecked(False)
		self.chkConvDropout.setChecked(False)
		self.txtConvKeepRate.setText('')
		self.txtPoolName.setText('')
		self.txtPoolKernelSize.setText('')
		self.txtPoolStride.setText('')
		self.cbxPoolPadding.setCurrentIndex(0)
		self.chkPoolNorm.setChecked(False)
		self.chkPoolDropout.setChecked(False)
		self.txtPoolKeepRate.setText('')
		self.txtFCName.setText('')
		self.cbxFCActFunction.setCurrentIndex(0)
		self.txtFCNumOutputNodes.setText('')
		self.cbxFCWeightInit.setCurrentIndex(0)
		self.cbxFCBiasInit.setCurrentIndex(0)
		self.txtFCStdDev.setText('')
		self.txtFCCBiasVal.setText('')
		self.chkFCNorm.setChecked(False)
		self.chkFCDropout.setChecked(False)
		self.txtFCKeepRate.setText('')
		self.txtOutputName.setText('')
		self.cbxOutputActFunction.setCurrentIndex(0)
		self.cbxOutputWeightInit.setCurrentIndex(0)
		self.cbxOutputBiasVal.setCurrentIndex(0)
		self.txtOutputStdDev.setText('')
		self.txtOutputConstantVal.setText('')

	def deleteModelButtonClicked(self):
		self.newModel = []
		self.lstModel.clear()

	def deleteLastLayerButtonClicked(self):
		try:
			self.newModel.pop(-1)
			self.lstModel.takeItem(len(self.newModel))
		except:
			self.txtOutputModelLog.append("No More Layers to Delete!")

	def biasInitComboBoxChanged(self):
		print("made it")
		if self.cbxConvBiasInit.currentIndex() == 0 or self.cbxConvBiasInit.currentIndex() == 1:
			self.lblConvBiasVal.setText('Std Dev of Weights')
		elif self.cbxConvBiasInit.currentIndex() == 2:
			self.lblConvBiasVal.setText('')
			self.txtConvBiasVal.setFixedWidth(0)

		
	def changeModelLoadPath(self):
		fname = QFileDialog.getOpenFileName(self, 'Open file', '/home', "XML Files (*.xml)")
		self.txtLoadModel.setText(str(fname[0])) 

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
		if self.radCIFAR10.isChecked():
			return 'CIFAR10'
		elif self.radMNIST.isChecked():
			return 'MNIST'

	def l2RegularizationCheckBoxStateChanged(self):
		if self.chkL2Reg.isChecked():
			self.txtL2Beta.setEnabled(True)
		else: self.txtL2Beta.setEnabled(False)

	def checkpointCheckBoxStateChange(self):
		if self.cbxSavePath.isChecked():
			self.txtSavePath.setText(self.txtLoadCheckpoints.text())

	def dropoutCheckBoxStateChanged(self):
		if self.chkConvDropout.isChecked():
			self.txtConvKeepRate.setEnabled(True)
		else: self.txtConvKeepRate.setEnabled(False)

		if self.chkPoolDropout.isChecked():
			self.txtPoolKeepRate.setEnabled(True)
		else: self.txtPoolKeepRate.setEnabled(False)

		if self.chkFCDropout.isChecked():
			self.txtFCKeepRate.setEnabled(True)
		else: self.txtFCKeepRate.setEnabled(False)

	def createModelRadClicked(self, enabled):
		if enabled:
			self.tabPages.setCurrentIndex(1)
			self.radLoadModel.setChecked(True)
			
	def openTrainTab(self):
		self.tabPages.setCurrentIndex(0)

	def openDesignTab(self):
		self.tabPages.setCurrentIndex(1)

	def openVisualizationTab(self):
		self.tabPages.setCurrentIndex(2)

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
		self.btnCancelTraining.setEnabled(False)

	@pyqtSlot(float)
	def updateProgressBar(self, progress: float):
		self.prgTrainingProgress.setValue(progress)

	@pyqtSlot(list)
	def showModelDetails(self, layers: list):
		for e in layers:
			self.txtOutputLog.append("Layer Name: {0}".format(e.layerName))
			if e.layerType == 'Convolution':
				self.txtOutputLog.append('Convolution Layer')
				self.txtOutputLog.append("Num Output Filters {0}".format(e.numOutputFilters))
				self.txtOutputLog.append("Kernel Size: [1,{0},{1},1], Stride: [1,{2},{3},1]".format(e.kernelSize, e.kernelSize, e.stride, e.stride))
				self.txtOutputLog.append("Activation Function: {0}".format(e.actFunction))  
				self.txtOutputLog.append("Padding: {0}".format(e.padding))  
				self.txtOutputLog.append("Normalize: {0}".format(e.normalize))  
				self.txtOutputLog.append("Dropout: {0}, Keep Rate {1} \n".format(e.dropout, e.keepRate)) 
			elif e.layerType == 'Max Pool':
				self.txtOutputLog.append('Max Pooling Layer')
				self.txtOutputLog.append("Kernel Size: [1,{0},{1},1], Stride: [1,{2},{3},1]".format(e.kernelSize, e.kernelSize, e.stride, e.stride))
				self.txtOutputLog.append("Padding: {0}".format(e.padding))  
				self.txtOutputLog.append("Normalize: {0}".format(e.normalize))  
				self.txtOutputLog.append("Dropout: {0}, Keep Rate {1} \n".format(e.dropout, e.keepRate)) 
			elif e.layerType == 'Dense':
				self.txtOutputLog.append('Dense Layer')
				self.txtOutputLog.append("Num Output Nodes {0}".format(e.numOutputNodes))
				self.txtOutputLog.append("Activation Function: {0}".format(e.actFunction))  
				self.txtOutputLog.append("Normalize: {0}".format(e.normalize))  
				self.txtOutputLog.append("Dropout: {0}, Keep Rate {1} \n".format(e.dropout, e.keepRate)) 
			elif e.layerType == 'Output':
				self.txtOutputLog.append('Output Layer')
				self.txtOutputLog.append("Activation Function: {0}".format(e.actFunction))  
		self.txtOutputLog.append('\n ------------------------------------------------------------------------------------------------------------- \n')

 

	@pyqtSlot(object, bool, int)
	def updateConfusionPlot(self, confusion: object, training: bool, epoch: int):
		plt.close()

		norm_conf = []

		for i in confusion:
			a = 0
			tmp_arr = []
			a = sum(i, 0)

			for j in i:
				tmp_arr.append(float(j)/float(a))
			norm_conf.append(tmp_arr)

		self.fig = plt.figure()
		self.axes = self.fig.add_subplot(111)

		self.axes.set_aspect(1)
			
		res = self.axes.imshow(np.array(norm_conf), cmap=plt.cm.jet, interpolation='nearest')
		
		width, height = confusion.shape

		for x in range(width):
			for y in range(height):
				self.axes.annotate(str(confusion[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

		cb = self.fig.colorbar(res)

		alphabet = '0123456789'

		# set axes to 0-9
		plt.xticks(range(width), alphabet[:width])
		plt.yticks(range(height), alphabet[:height])

		if training == True:
			plt.savefig('training_confusion_matrix.png', format='png')
			pixMap = QPixmap("training_confusion_matrix.png")
			self.lblTrainingConfusionMat.setPixmap(pixMap)
			self.lblBatchConf.setText("Batch {0}".format(epoch))
		else:
			plt.savefig('testing_confusion_matrix.png', format='png')
			pixMap = QPixmap("testing_confusion_matrix.png")
			self.lblTestingConfusionMat.setPixmap(pixMap)

		plt.clf()



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
