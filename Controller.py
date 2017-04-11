import sys
import tensorflow as tf
import numpy as np
import pickle
import datetime
import cv2
import xml.etree.ElementTree as ET
import matplotlib as mpl
#mpl.use('Agg')
mpl.use('Qt5Agg')

import CNN_GUI_V14 as design
import input_data as data
import xml_parser as pp
import Layer as l
import layer_factory as factory

from functools import partial
from math import ceil, sqrt
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QApplication, QMessageBox, QFileDialog, QSizePolicy, QListWidgetItem, QVBoxLayout
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
#plt.rcParams['axes.facecolor'] = 'peachpuff'
#np.set_printoptions(threshold=np.inf)

class Worker(QObject):

	# signals for updating GUI elements
	epoch_progress = pyqtSignal(float)
	test_set_accuracy = pyqtSignal(float)
	log_message = pyqtSignal(str)
	network_model = pyqtSignal(list)

	# signals for updating/creating visualizations
	train_valid_error = pyqtSignal(float, int)
	train_valid_acc = pyqtSignal(float, float, int)
	batch_loss = pyqtSignal(float, int)
	batch_acc = pyqtSignal(float, int)
	confusion_mat = pyqtSignal(bool, int)
	network_weights_outputs = pyqtSignal(list, list)

	# signal to inform main thread that worker thread has finished work
	work_complete = pyqtSignal()

	def __init__(self, data_set: str, validation: bool, num_epochs: int, batch_size: int, learning_rate: float, momentum: float, optimizer: int, normalize: bool, l2_reg: bool, beta: float, save_directory: str, save_interval: int, model_path: str, run_time: bool):
		super().__init__()
		self.end_training = False
		self.data_set = data_set
		self.regression = False
		self.validation = validation
		self.num_epochs = num_epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.optimizer = optimizer
		self.normalize = normalize
		self.l2_reg = l2_reg
		self.beta = beta
		self.save_directory = save_directory
		self.save_interval = save_interval
		self.model_path = model_path
		self.run_time = run_time
		
	@pyqtSlot()
	def work(self):
		self.train_network(self.data_set, self.validation, self.num_epochs, self.batch_size, self.learning_rate, self.momentum, self.optimizer, 
			self.normalize, self.l2_reg, self.beta, self.save_directory, self.save_interval, self.model_path, self.run_time)

	# function that calculates and returns RMSE of current batch
	def calc_accuracy(self, predictions, labels, verbose=True):

		# Convert back to degree
		predictions_degree = predictions * 180
		predictions_degree -= 90
		labels_degree = labels * 180
		labels_degree -= 90

		RMSE_pitch = np.sum(np.square(predictions_degree - labels_degree), dtype=np.float32) * 1 / predictions.shape[0]
		RMSE_pitch = np.sqrt(RMSE_pitch)
		RMSE_std = np.std(np.sqrt(np.square(predictions_degree - labels_degree)), dtype=np.float32)
		# MAE = Mean Absolute Error
		MAE_pitch = np.sum(np.absolute(predictions_degree - labels_degree), dtype=np.float32) * 1 / predictions.shape[0]
		MAE_std = np.std(np.absolute(predictions_degree - labels_degree), dtype=np.float32)

		if (verbose == True):
			print("==============================")            
			print("RMSE mean: " + str(RMSE_pitch) + " degree")
			print("RMSE std: " + str(RMSE_std) + " degree")
			print("MAE mean: " + str(MAE_pitch) + " degree")
			print("MAE std: " + str(MAE_std) + " degree")
			print("==============================")

		return RMSE_pitch


	def train_network(self, data_set, validation, num_epochs, batch_size, learning_rate, momentum, learning_algo, normalize, l2_reg, beta, save_path, save_interval, model_path, run_time):

		# load selected data set
		if (data_set == 'CIFAR10'):
			training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_CIFAR_10(validation)
		if (data_set == 'MNIST'):
			training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_MNIST(validation)
		if (data_set == 'PrimaHeadPose'):
			training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_prima_head_pose()
			self.regression = True

		# normalize data, if user requires
		if normalize == True:
			training_set -= 127 
			testing_set -= 127
			if validation == True:
				validation_set -= 127

		try:
			# get array of layers from XML file
			layers = pp.get_layers(model_path)
		except:
			self.log_message.emit('Error reading XML file')
			return 0

		self.network_model.emit(layers)

		self.log_message.emit('Initialising Tensorflow Variables... \n')
		
		graph = tf.Graph()
		with graph.as_default():

			# define placeholder variables
			x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
			y = tf.placeholder(tf.float32, shape=(None, num_classes))
			keep_prob = tf.placeholder(tf.float32)

			global_step = tf.Variable(0, trainable=False)


			if self.regression == False:
				# stores the class integer values 
				class_labels = tf.argmax(y, dimension=1)
			else: 
				angle_labels = y

			# array to hold tensorflow layers
			network_layers = []

			def CNN_Model(data, _dropout=1.0):

				# instantiate layer factory
				layer_factory = factory.layer_factory(num_classes, num_channels, layers)

				# set input to first layer as x (input data)
				layer = x

				input_dimension = image_size

				# create as many layers as are stored in XML file
				for e in range(len(layers)):
					layer, input_dimension = layer_factory.create_layer(layer, layers[e], input_dimension)
					network_layers.append(layer)

				# return last element of layers array (output layer)
				return network_layers[-1]


			model_output = CNN_Model(x, keep_prob)

			# define loss term dependent upon which data set is in use.. Prima = regression
			if self.regression == False:
				cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model_output, labels=y)
				loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
			else: loss = tf.nn.l2_loss(model_output - angle_labels)

			# add l2 regularization if user has specified
			if l2_reg == True:
				# for all trainable variables in graph
				for e in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
					# if a weights variable (excluding biases), add regularization term
					if 'weights' in e.name:
						loss += (beta * tf.nn.l2_loss(e))
			

			# consider adding option for learning rates like this 
			#learning_rate = tf.train.exponential_decay(0.0125, global_step, 15000, 0.1, staircase=True)

			if (learning_algo == 0):
				optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
				self.log_message.emit('Optimizer: Gradient Descent')
			elif (learning_algo == 1):
				optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
				self.log_message.emit('Optimizer: Adam')
			elif (learning_algo == 2):
				optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
				self.log_message.emit('Optimizer: Ada Grad')
			elif (learning_algo == 3):
				optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step)
				self.log_message.emit('Optimizer: Ada Delta')
			elif (learning_algo == 4):
				optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step=global_step)
				self.log_message.emit('Optimizer: Momentum')


			if self.regression == False:
				# convert one hot encoded array to single prediction value
				network_pred_class = tf.argmax(model_output, dimension=1)
				correct_prediction = tf.equal(network_pred_class, class_labels)
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			else: 
				network_pred_angle = model_output


			# create saver object to save/restore checkpoints
			saver = tf.train.Saver()

			# begin session
			with tf.Session(graph=graph) as session:

				 # Use TensorFlow to find the latest checkpoint - if any.
				last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)

				self.log_message.emit('Trying to restore last checkpoint ...')
				try:
					# Try and load the data in the checkpoint.
					saver.restore(session, save_path=last_chk_path)
					self.log_message.emit('Restored checkpoint from: {} '.format(last_chk_path))
				except:
					self.log_message.emit('Failed to restore checkpoint. Initializing variables instead.')
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

						# feed batch through network                        this shouldnt be here?
						feed_dict = {x: batch_data, y: batch_labels, keep_prob: 0.5}

						if self.regression == False:
							_, l, acc, batch_pred_class, batch_classes = session.run([optimizer, loss, accuracy, network_pred_class, class_labels], feed_dict=feed_dict)
						else: 
							_, l, batch_pred_angles = session.run([optimizer, loss, model_output], feed_dict=feed_dict)


						# print information to output log
						if (epoch % 100 == 0):
							self.log_message.emit('')
							self.log_message.emit('Loss at epoch: {} of {} is {}'.format(epoch, str(num_epochs), l))
							self.log_message.emit('Global Step: {}'.format(str(global_step.eval())))
							self.log_message.emit('Learning Rate: {}'.format(str(learning_rate)))
							self.log_message.emit('Minibatch size: {}'.format(str(batch_labels.shape)))

							if self.regression == False:
								self.log_message.emit('Batch Accuracy = {}'.format(str(acc)))
								self.batch_acc.emit(acc, epoch)
							else: 
								RMSE = self.calc_accuracy(batch_pred_angles, batch_labels)
								self.log_message.emit('Batch RMSE = {}'.format(str(RMSE)))
								self.batch_acc.emit(RMSE, epoch)

							
							# emit batch loss for GUI visualization
							self.batch_loss.emit(l, epoch)
							

							# if user has chosen to include run time visualizations
							if run_time == True:

								if self.regression == False:
									# create confusion matrix for current batch
									batch_confusion = tf.contrib.metrics.confusion_matrix(batch_classes, batch_pred_class).eval()
									self.create_confusion_matrix(batch_confusion, False)
									self.confusion_mat.emit(True, epoch)

								if validation == True:

									num_training_data = len(training_labels)
									data_per_batch = 10000
									num_batches = num_training_data / data_per_batch

									for i in range(int(num_batches)):
									   batch_data = training_set[i*data_per_batch:i+1*data_per_batch]
									   batch_labels = training_labels[i*data_per_batch:i+1*data_per_batch]
									   train_accuracy = session.run(accuracy, feed_dict={x: batch_data, y: batch_labels, keep_prob: 1.0})

									# run validation set at current batch
									valid_accuracy = session.run(accuracy, feed_dict={x: validation_set, y: validation_labels, keep_prob: 1.0})

									self.train_valid_acc.emit(train_accuracy, valid_accuracy, epoch)


						# calculate progress as percentage and emit signal to GUI to update
						epoch_prog = (epoch / num_epochs) * 100
						self.epoch_progress.emit(epoch_prog)

						# check save interval to avoid divide by zero
						if save_interval != 0:
							# save at interval define by user
							if (epoch % save_interval == 0):
								saver.save(sess=session, save_path=save_path, global_step=global_step)
								self.log_message.emit('Saved Checkpoint \n')

				self.log_message.emit('\nTraining Complete')
				# save network state when complete
				saver.save(sess=session, save_path=save_path, global_step=global_step)

				self.log_message.emit('Saved Checkpoint')

				# set progress bar to 100% (if training was not interrupted)
				if (self.end_training == False):
					self.epoch_progress.emit(100)

				self.log_message.emit('\nEvaluating Test Set...')

				feed_dict = {x: testing_set, y:testing_labels, keep_prob: 1.0}

				# run test set through network
				if self.regression == False:
					# if classification, grab accuracy from tensorflow graph
					test_acc, testing_pred_class, testing_classes = session.run([accuracy, network_pred_class, class_labels], feed_dict=feed_dict)
				else: 
					# otherwise calculate RMSE using function
					batch_pred_angles = session.run(model_output, feed_dict=feed_dict)
					test_acc = self.calc_accuracy(batch_pred_angles, testing_labels)


				self.log_message.emit('Test Set Evaluated \n')
				
				# send test accuracy to GUI
				self.test_set_accuracy.emit(test_acc)

				self.log_message.emit('Loading Visualizations...')

				# lists to hold file names of each plot
				layer_weights_file_names = []
				layer_outputs_file_names = []

				# create convolution weight plots for each convolution layer
				for e in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
					if 'Cweights' in e.name:
						file_name = self.plot_conv_weights(weights=e, name=e.name, session=session)
						layer_weights_file_names.append(file_name)

				# grab random image from test set
				random = np.random.randint(0, testing_set.shape[0])
				image = testing_set[random]
				layer_count = 0
				# create convolution output plots for each convolution layer
				for layer in network_layers:
					if layers[layer_count].layer_type == 'Convolution':
						file_name = self.plot_conv_layer(layer=layer, name=layers[layer_count].layer_name, image=image, session=session, x=x)
						layer_outputs_file_names.append(file_name)
					layer_count += 1


				# send file paths for newly created plots to GUI to load on screen
				self.network_weights_outputs.emit(layer_weights_file_names, layer_outputs_file_names)

				if self.regression == False:
					# create confusion matrix from predicted and actual classes
					test_set_confusion = tf.contrib.metrics.confusion_matrix(testing_classes, testing_pred_class).eval()
					self.create_confusion_matrix(test_set_confusion, False)
					self.confusion_mat.emit(False, 0)

				self.log_message.emit('Visualizations Loaded\n')

				# signal that thread has completed
				self.work_complete.emit()


	def cancel_training(self):
		self.end_training = True

	def plot_conv_layer(self, layer, name, image, session, x):

		feed_dict = {x: [image]}

		# Calculate and retrieve the output values of the layer
		# when inputting that image.
		values = session.run(layer, feed_dict=feed_dict)

		values_min = np.min(values)
		values_max = np.max(values)

		# Number of filters used in the conv. layer.
		num_filters = values.shape[3]

		# Number of grids to plot.
		# Rounded-up, square-root of the number of filters.
		num_grids = ceil(sqrt(num_filters))
		
		# Create figure with a grid of sub-plots.
		fig, axes = plt.subplots(num_grids, num_grids)

		# Plot the output images of all the filters.
		for i, ax in enumerate(axes.flat):
			# Only plot the images for valid filters.
			if i < num_filters:
				# Get the output image of using the i'th filter.
				# See new_conv_layer() for details on the format
				# of this 4-dim tensor.
				img = values[0, :, :, i]

			# Plot image.
			ax.imshow(img, vmin=values_min, vmax=values_max, interpolation='nearest', cmap='binary')
			
			# Remove ticks from the plot.
			ax.set_xticks([])
			ax.set_yticks([])

		file_name = name + "_output.png"
		plt.savefig(file_name, format='png')
		plt.close()
		return file_name

	def plot_conv_weights(self, weights, name, session, input_channel=0):
		w = session.run(weights)

		w_min = np.min(w)
		w_max = np.max(w)
		abs_max = max(abs(w_min), abs(w_max))

		# Number of filters used in the conv. layer.
		num_filters = w.shape[3]

		# Number of grids to plot.
		# Rounded-up, square-root of the number of filters.
		num_grids = ceil(sqrt(num_filters))
		
		# Create figure with a grid of sub-plots.
		fig, axes = plt.subplots(num_grids, num_grids)

		# Plot all the filter-weights.
		for i, ax in enumerate(axes.flat):
			# Only plot the valid filter-weights.
			if i < num_filters:
				# Get the weights for the i'th filter of the input channel.
				# See new_conv_layer() for details on the format
				# of this 4-dim tensor.
				img = w[:, :, input_channel, i]

				# Plot image.
				ax.imshow(img, vmin=-abs_max, vmax=abs_max, interpolation='nearest', cmap='seismic')
			
			# Remove ticks from the plot.
			ax.set_xticks([])
			ax.set_yticks([])

		name = name[:-2]
		file_name = name + ".png"
		plt.savefig(file_name, format='png')
		plt.close()
		return file_name

	def create_confusion_matrix(self, confusion, training):

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
		else:
			plt.savefig('testing_confusion_matrix.png', format='png')


class MyMplCanvas(FigureCanvas):
	"""Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
	def __init__(self, parent=None, width=5, height=4, dpi=100, title='title', xAxisTitle='x', yAxisTitle='y'):
		self.title = title
		self.xAxisTitle = xAxisTitle
		self.yAxisTitle = yAxisTitle

		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = self.fig.add_subplot(111)
		self.fig.suptitle(title)

		self.axes.set_xlabel(self.xAxisTitle)
		self.axes.set_ylabel(self.yAxisTitle)

		self.axes.grid(True)

		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)

		FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)


class DynamicGraph(MyMplCanvas):
	"""A canvas that updates itself every secoSnd with a new plot."""
	def __init__(self, *args, **kwargs):
		MyMplCanvas.__init__(self, *args, **kwargs)
		self.errors = []
		self.epochs = []

	@pyqtSlot(float, int)
	def update_figure(self, error: float, epoch: int):

		if epoch == 0:
			self.axes.clear()
			self.axes.set_xlabel(self.xAxisTitle)
			self.axes.set_ylabel(self.yAxisTitle)
			self.axes.grid(True)
			self.errors = []
			self.epochs = []

		# add current values to array
		self.errors.append(error)
		self.epochs.append(epoch)

		# plot new, extended graph
		self.axes.plot(self.epochs, self.errors, 'r')
		#self.axes.fill_between(self.epochs, 0, self.errors, alpha=0.5, facecolor='r')
		
		#self.axes.grid(True)
		self.draw()

class DynamicDoubleGraph(MyMplCanvas):
	"""A canvas that updates itself every secoSnd with a new plot."""
	def __init__(self, *args, **kwargs):
		MyMplCanvas.__init__(self, *args, **kwargs)
		self.train_errors = []
		self.valid_errors = []
		self.epochs = []

	@pyqtSlot(float, float, int)
	def update_figure(self, train_error: float, valid_error: float, epoch: int):

		if epoch == 0:
			self.axes.clear()
			self.axes.set_xlabel(self.xAxisTitle)
			self.axes.set_ylabel(self.yAxisTitle)
			self.axes.grid(True)
			self.train_errors = []
			self.valid_errors = []
			self.epochs = []

		#ax2 = self.fig.add_subplot(212, sharex=self.axes)

		# add current values to array
		self.train_errors.append(train_error)
		self.valid_errors.append(valid_error)
		self.epochs.append(epoch)

		# plot new, extended graph
		self.axes.plot(self.epochs, self.train_errors, 'r', label="train")
		self.axes.plot(self.epochs, self.valid_errors, 'blue', label="valid")
		self.draw()

class CNNApp(QMainWindow, design.Ui_MainWindow):

	# signal to background thread that the user has requested training to cancel
	end_train = pyqtSignal()

	def __init__(self):  
		super().__init__()
		self.setupUi(self)

		# initialise threads array, if multiple required
		self.__threads = None
		self.new_model = []

		# initialise GUI elements
		self.btnCancelTraining.setEnabled(False)
		self.txtConvKeepRate.setEnabled(False)
		self.txtPoolKeepRate.setEnabled(False)
		self.txtFCKeepRate.setEnabled(False)
		self.btnCreateModel.setEnabled(False)
		self.txtL2Beta.setEnabled(False)

		# for quickness. TEMPORRARY
		self.txtLoadModel.setText('C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/Tensorflow-Final-Year-Project/Models/MNIST.xml')

		# navigational buttons
		self.actionTrain.triggered.connect(partial(self.open_tab, index=0))
		self.actionDesign.triggered.connect(partial(self.open_tab, index=1))
		self.actionVisualizations.triggered.connect(partial(self.open_tab, index=2))
		self.actionSettings.triggered.connect(partial(self.open_tab, index=3))
		self.actionExit.triggered.connect(self.close)

		self.cbxSavePath.stateChanged.connect(self.checkpoint_checkbox_state_changed)
		self.radCreateModel.toggled.connect(self.create_model_rad_clicked)

		# events for checking/unchecking dropout when creating model
		self.chkConvDropout.stateChanged.connect(partial(self.dropout_checkbox_state_changed, dropout_checkbox=self.chkConvDropout, keep_rate_text_field=self.txtConvKeepRate))
		self.chkPoolDropout.stateChanged.connect(partial(self.dropout_checkbox_state_changed, dropout_checkbox=self.chkPoolDropout, keep_rate_text_field=self.txtPoolKeepRate))
		self.chkFCDropout.stateChanged.connect(partial(self.dropout_checkbox_state_changed, dropout_checkbox=self.chkFCDropout, keep_rate_text_field=self.txtFCKeepRate))

		self.cbxOptimizer.currentIndexChanged.connect(self.optimizer_combo_box_changed)
		self.chkL2Reg.stateChanged.connect(self.l2_reg_checkbox_state_changed)

		self.cbxConvBiasInit.currentIndexChanged.connect(partial(self.bias_init_combo_changed, bias_init_combobox=self.cbxConvBiasInit, bias_val_text_field= self.txtConvBiasVal, bias_val_label= self.lblConvBiasVal))
		self.cbxFCBiasInit.currentIndexChanged.connect(partial(self.bias_init_combo_changed, bias_init_combobox=self.cbxFCBiasInit, bias_val_text_field= self.txtFCBiasVal, bias_val_label= self.lblFCBiasVal))
		self.cbxOutputBiasInit.currentIndexChanged.connect(partial(self.bias_init_combo_changed, bias_init_combobox=self.cbxOutputBiasInit, bias_val_text_field= self.txtOutputBiasVal, bias_val_label= self.lblOutputBiasVal))

		# buttons to change file paths 
		self.btnChangeSavePath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtSavePath))
		self.btnChangeLoadCheckpoints.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtLoadCheckpoints, disable=True))
		self.btnChangeModelPath.clicked.connect(self.change_file_path)
		self.btnChangeModelSavePath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtModelSavePath))
		
		# train/cancel training buttons
		self.btnTrainNetwork.clicked.connect(self.train_button_clicked)
		self.btnCancelTraining.clicked.connect(self.cancel_train)

		self.radMNIST.toggled.connect(partial(self.data_set_rad_state_changed, False))
		self.radCIFAR10.toggled.connect(partial(self.data_set_rad_state_changed, False))
		self.radPrimaHeadPose.toggled.connect(partial(self.data_set_rad_state_changed, True))

		# create new model buttons
		self.btnAddConvLayer.clicked.connect(self.create_conv_layer_button_clicked)
		self.btnAddMaxPool.clicked.connect(self.create_pooling_layer_button_clicked)
		self.btnAddFullyConn.clicked.connect(self.create_full_conn_layer_button_clicked)
		self.btnAddOutput.clicked.connect(self.create_output_layer_button_clicked)
		self.btnValidateNetwork.clicked.connect(self.validate_model_button_clicked)
		self.btnCreateModel.clicked.connect(self.create_model_button_clicked)
		self.btnDeleteModel.clicked.connect(self.delete_model_button_clicked)
		self.btnDeleteLayer.clicked.connect(self.delete_last_layer_button_clicked)

		# clear output logs
		self.btnClearLog.clicked.connect(self.clear_output_log)
		self.btnClearModelLog.clicked.connect(self.clear_output_model_log)

		# create graph instances
		self.batch_loss_graph = DynamicGraph(self.grphBatchLoss, width=5, height=4, dpi=100, title='Loss Over Epochs', xAxisTitle='Epoch', yAxisTitle='Loss')
		self.batch_acc_graph = DynamicGraph(self.grphBatchAcc, width=5, height=4, dpi=100, title='Accuracy Over Epochs', xAxisTitle='Epoch', yAxisTitle='Accuracy')
		self.train_valid_accuracy = DynamicDoubleGraph(self.grphTrainValidAcc, width=5, height=4, dpi=100, title='Training / Validation Accuracy Over Epochs', xAxisTitle='Epoch', yAxisTitle='Accuracy')

		
	def train_button_clicked(self):
		try:
			'''
			num_epochs = int(self.txtNumEpochs.text())
			batch_size = int(self.txtBatchSize.text())
			learning_rate = float(self.txtLearningRate.text())
			optimizer = int(self.cbxOptimizer.currentIndex())
			'''
			num_epochs = 500
			batch_size = 64
			learning_rate = 0.0005
			optimizer = 1
			
			# obtain model file path and checkpoint save path from user text fields
			model_path = self.txtLoadModel.text()
			save_path = self.txtSavePath.text()

			if self.chkValidationSet.isChecked():
				validation = True
			else: validation = False

			if self.cbxOptimizer.currentIndex == 4:
				momentum = float(self.txtMomentum.text())
			else: momentum = 0

			if self.chkNormalize.isChecked():
				normalize = True
			else: normalize = False

			if self.chkL2Reg.isChecked():
				l2_reg = True
				beta = float(self.txtL2Beta.text())
			else:
				l2_reg = False
				beta = 0

			if self.chkVisualizations.isChecked():
				run_time = True
			else: run_time = False

			if self.cbxSaveInterval.currentIndex() == 0:
				save_interval = 0
			else: save_interval = int(self.cbxSaveInterval.currentText())
	
		except ValueError:
			self.txtOutputLog.append('Number of Epochs, Batch Size and Learning Rate, Momentum and L2 Beta must be a Numerical Value!')
		else:
			# initialise threads array
			self.__threads = []

			# update GUI
			self.btnTrainNetwork.setDisabled(True)
			self.btnCancelTraining.setEnabled(True)
			self.prgTrainingProgress.setValue(0)

			# create worker object and thread
			worker = Worker(self.current_data_set(), validation, num_epochs, batch_size, learning_rate, momentum, optimizer, normalize, l2_reg, beta, save_path, save_interval, model_path, run_time)
			thread = QThread()

			# connect cancel button in main thread to background thread
			self.end_train.connect(worker.cancel_training) # THIS WOULD NOT WORK A FEW LINES LOWER!!!!

			# store reference to objects so they are not garbage collected
			self.__threads.append((thread, worker))
			worker.moveToThread(thread)

			# set connections from background thread to main thread for updating GUI elements
			worker.test_set_accuracy.connect(self.update_test_set_accuracy)
			worker.epoch_progress.connect(self.update_progress_bar)
			worker.log_message.connect(self.txtOutputLog.append)
			worker.network_model.connect(self.show_model_details)
			worker.work_complete.connect(self.abort_workers)

			# set connections from background thread to create/update visualizations on GUI
			worker.batch_loss.connect(self.batch_loss_graph.update_figure)
			worker.batch_acc.connect(self.batch_acc_graph.update_figure)
			worker.train_valid_acc.connect(self.train_valid_accuracy.update_figure)
			worker.confusion_mat.connect(self.update_confusion_plot)
			worker.network_weights_outputs.connect(self.embed_network_weights_outputs)

			# connect and start thread
			thread.started.connect(worker.work)
			thread.start()

	def create_conv_layer_button_clicked(self):

		try:
			conv_layer_name = self.txtConvName.text()
			conv_kernel_size = int(self.txtConvKernelSize.text())
			conv_stride = int(self.txtConvStride.text())
			num_output_filters = int(self.txtConvOutputFilters.text())
			act_function = self.return_act_function(self.cbxConvActFunction)
			weight_init, weight_std_dev = self.return_weight_init(self.cbxConvWeightInit, self.txtConvStdDev)
			bias_init, bias_val = self.return_bias_init(self.cbxConvBiasInit, self.txtConvBiasVal)
			padding = self.return_padding(self.cbxConvPadding)
			normalize = self.return_normalize(self.chkConvNorm)
			dropout, keep_rate = self.return_dropout(self.chkConvDropout, self.txtConvKeepRate)

			layer = l.ConvLayer(conv_layer_name, conv_kernel_size, conv_stride, act_function, num_output_filters, weight_init, weight_std_dev, bias_init, bias_val, padding, normalize, dropout, keep_rate)
			self.new_model.append(layer)

			item = QListWidgetItem(("Convolution, Num Output Filters {}, Kernel Size: {}, Stride: {}, Activation Function: {}, Padding: {}, Normalize: {}, Dropout: {}, Keep Rate: {}").format(layer.num_output_filters, layer.kernel_size, layer.stride, layer.act_function, layer.padding, layer.normalize, layer.dropout, layer.keep_rate))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	def create_pooling_layer_button_clicked(self):
		try:    
			pool_layer_name = self.txtPoolName.text()
			pool_kernel_size = int(self.txtPoolKernelSize.text())
			pool_stride = int(self.txtPoolStride.text())
			padding = self.return_padding(self.cbxPoolPadding)
			normalize = self.return_normalize(self.chkPoolNorm)
			dropout, keep_rate = self.return_dropout(self.chkPoolDropout, self.txtPoolKeepRate)

			layer = l.MaxPoolingLayer(pool_layer_name, pool_kernel_size, pool_stride, padding, normalize, dropout, keep_rate)
			self.new_model.append(layer)

			item = QListWidgetItem(("Max Pool, Kernel Size: {}, Stride: {}, Padding: {}, Normalize: {}, Dropout: {}, Keep Rate: {}").format(layer.kernel_size, layer.stride, layer.padding, layer.normalize, layer.dropout, layer.keep_rate))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	def create_full_conn_layer_button_clicked(self):

		try:
			FC_layer_name = self.txtFCName.text()
			num_output_nodes = int(self.txtFCNumOutputNodes.text())
			act_function = self.return_act_function(self.cbxFCActFunction)
			weight_init, weight_std_dev = self.return_weight_init(self.cbxFCWeightInit, self.txtFCStdDev)
			bias_init, bias_val = self.return_bias_init(self.cbxFCBiasInit, self.txtFCBiasVal)
			normalize = self.return_normalize(self.chkFCNorm)
			dropout, keep_rate = self.return_dropout(self.chkFCDropout, self.txtFCKeepRate)

			layer = l.FullyConnectedLayer(FC_layer_name, act_function, num_output_nodes, weight_init, weight_std_dev, bias_init, bias_val,  normalize, dropout, keep_rate)
			self.new_model.append(layer)

			item = QListWidgetItem(("Fully Connected,  Num Output Nodes: {}, Activation Function: {}, Normalize: {}, Dropout: {}, Keep Rate: {}").format(layer.num_output_nodes, layer.act_function, layer.normalize, layer.dropout, layer.keep_rate))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	def create_output_layer_button_clicked(self):
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

			weight_init, weight_std_dev = self.return_weight_init(self.cbxOutputWeightInit, self.txtOutputStdDev)
			bias_init, bias_val = self.return_bias_init(self.cbxOutputBiasInit, self.txtOutputBiasVal)

			layer = l.OutputLayer(output_layer_name, act_function, weight_init, weight_std_dev, bias_init, bias_val)
			self.new_model.append(layer)

			item = QListWidgetItem(("Output, Activation Function: {}").format(layer.act_function))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	def return_act_function(self, comboBox):
		if comboBox.currentIndex() == 0:
			act_function = 'ReLu'
		elif comboBox.currentIndex() == 1:
			act_function = 'Sigmoid'
		elif comboBox.currentIndex() == 2:
			act_function = 'Tanh'
		return act_function

	def return_weight_init(self, comboBox, textField):
		if comboBox.currentIndex() == 0:
			weight_init = "Random Normal"
			weight_std_dev = float(textField.text())
		elif comboBox.currentIndex() == 1:
			weight_init = "Truncated Normal"
			weight_std_dev = float(textField.text())
		return weight_init, weight_std_dev

	def return_bias_init(self, comboBox, textField):
		if comboBox.currentIndex() == 0:
			bias_init = "Random Normal"
			bias_val = float(textField.text())
		elif comboBox.currentIndex() == 1:
			bias_init = "Truncated Normal"
			bias_val = float(textField.text())
		elif comboBox.currentIndex() == 2:
			bias_init = "Zeros"
			bias_val = 0
		elif comboBox.currentIndex() == 3:
			bias_init = "Constant"
			bias_val = float(textField.text())
		return bias_init, bias_val

	def return_padding(self, comboBox):
		if comboBox.currentIndex() == 0:
			padding = 'SAME'
		elif comboBox.currentIndex() == 1:
			padding = 'VALID'   
		return padding  

	def return_normalize(self, checkBox):
		if checkBox.isChecked():
			normalize = True
		else: normalize = False
		return normalize

	def return_dropout(self, checkBox, textField):
		if checkBox.isChecked():
			dropout = True
			keep_rate = float(textField.text())
		else: 
			dropout = False
			keep_rate = 1.0
		return dropout, keep_rate

	def validate_model_button_clicked(self):

		if not self.new_model:
			self.txtOutputModelLog.append('No Layers Added')
			return

		if not self.new_model[0].layer_type == 'Convolution':
			self.txtOutputModelLog.append('First layer must be Convolution layer')
			return

		if not self.new_model[-1].layer_type == 'Output':
			self.txtOutputModelLog.append('Final layer must be output layer')
			return

		self.txtOutputModelLog.append('Model Successfully Validated')
		self.btnCreateModel.setEnabled(True)

	def create_model_button_clicked(self):

		try:
			fileName = self.txtSaveModelAs.text()
			if not fileName:
				self.txtOutputModelLog.append("Please add name for new model")
				return False

			filePath = self.txtModelSavePath.text()

			success = pp.create_XML_model(self.new_model, fileName, filePath)

			if success == True:
				self.txtOutputModelLog.append("Success Writing XML File")
				self.reset_model_creation()
			else : 
				self.txtOutputModelLog.append("Error Writing XML File")
				return False
		except:
			self.txtOutputModelLog.append("Error Writing XML File")
		
	def reset_model_creation(self):
		self.delete_model_button_clicked()
		self.txtSaveModelAs.setText('')
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
		self.txtFCBiasVal.setText('')
		self.chkFCNorm.setChecked(False)
		self.chkFCDropout.setChecked(False)
		self.txtFCKeepRate.setText('')
		self.txtOutputName.setText('')
		self.cbxOutputActFunction.setCurrentIndex(0)
		self.cbxOutputWeightInit.setCurrentIndex(0)
		self.txtOutputBiasVal.setText('')
		self.txtOutputStdDev.setText('')
		self.txtOutputBiasVal.setText('')

	def delete_model_button_clicked(self):
		if not self.new_model:
			self.txtOutputModelLog.append("No Model to Delete!")
			return
		self.new_model = []
		self.lstModel.clear()

	def delete_last_layer_button_clicked(self):
		try:
			self.new_model.pop(-1)
			self.lstModel.takeItem(len(self.new_model))
		except:
			self.txtOutputModelLog.append("No More Layers to Delete!")

	def bias_init_combo_changed(self, bias_init_combobox, bias_val_text_field, bias_val_label):
		if bias_init_combobox.currentIndex() == 0 or bias_init_combobox.currentIndex() == 1:
			bias_val_label.setText('Std Dev of Weights:')
			bias_val_text_field.setFixedWidth(60)
		elif bias_init_combobox.currentIndex() == 2:
			bias_val_label.setText('')
			bias_val_text_field.setFixedWidth(0)
		elif bias_init_combobox.currentIndex() == 3:
			bias_val_label.setText('Value:')
			bias_val_text_field.setFixedWidth(60)

	def change_directory_path(self, path_text_field, disable=False):
		path = QFileDialog.getExistingDirectory(self, "Select Directory")
		path_text_field.setText(path + "/")
		if disable == True:
			self.cbxSavePath.setCheckState(False)

	def change_file_path(self):
		file = QFileDialog.getOpenFileName(self, 'Open file', '/home', "XML Files (*.xml)")
		self.txtLoadModel.setText(str(file[0])) 

	def data_set_rad_state_changed(self, prima_head_pose):
		if prima_head_pose == True:
			self.chkValidationSet.setFixedWidth(0)
			self.chkValidationSet.setChecked(False)
			self.lblTestAccuracy.setText('Test Set RMSE:')
		else: 
			self.chkValidationSet.setFixedWidth(110)
			self.lblTestAccuracy.setText('Test Set Accuracy:')

	def current_data_set(self):
		if self.radCIFAR10.isChecked():
			return 'CIFAR10'
		elif self.radMNIST.isChecked():
			return 'MNIST'
		elif self.radPrimaHeadPose.isChecked():
			return 'PrimaHeadPose'

	def optimizer_combo_box_changed(self):
		if self.cbxOptimizer.currentIndex() == 4:
			self.txtMomentum.setFixedWidth(60)
			self.lblMomentum.setFixedWidth(60)
		else:
			self.txtMomentum.setFixedWidth(0)
			self.lblMomentum.setFixedWidth(0)   

	def l2_reg_checkbox_state_changed(self):
		if self.chkL2Reg.isChecked():
			self.txtL2Beta.setEnabled(True)
		else: self.txtL2Beta.setEnabled(False)

	def checkpoint_checkbox_state_changed(self):
		if self.cbxSavePath.isChecked():
			self.txtSavePath.setText(self.txtLoadCheckpoints.text())

	def dropout_checkbox_state_changed(self, dropout_checkbox, keep_rate_text_field):
		if dropout_checkbox.isChecked():
			keep_rate_text_field.setEnabled(True)
		else: keep_rate_text_field.setEnabled(False)

	def create_model_rad_clicked(self, enabled):
		if enabled:
			self.tabPages.setCurrentIndex(1)
			self.radLoadModel.setChecked(True)
			
	def open_tab(self, index):
		self.tabPages.setCurrentIndex(index)

	def clear_output_log(self):
		self.txtOutputLog.setText('')

	def clear_output_model_log(self):
		self.txtOutputModelLog.setText('')

	def close_event(self, event):
		reply = QMessageBox.question(self, 'Message', "Are you sure you want to quit? All Unsaved Progress will be lost...", 
			QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
		if reply == QMessageBox.Yes:
			event.accept()
		else: event.ignore()
	

	@pyqtSlot(float)
	def update_test_set_accuracy(self, accuracy: float):
		self.txtTestAccuracy.setText(str(round(accuracy, 6)))
		self.btnCancelTraining.setEnabled(False)

	@pyqtSlot(float)
	def update_progress_bar(self, progress: float):
		self.prgTrainingProgress.setValue(progress)

	@pyqtSlot(list)
	def show_model_details(self, layers: list):
		for e in layers:
			self.txtOutputLog.append("Layer Name: {0}".format(e.layer_name))
			if e.layer_type == 'Convolution':
				self.txtOutputLog.append('Convolution Layer')
				self.txtOutputLog.append("Num Output Filters {0}".format(e.num_output_filters))
				self.txtOutputLog.append("Kernel Size: [1,{0},{1},1], Stride: [1,{2},{3},1]".format(e.kernel_size, e.kernel_size, e.stride, e.stride))
				self.txtOutputLog.append("Activation Function: {0}".format(e.act_function))  
				self.txtOutputLog.append("Padding: {0}".format(e.padding))  
				self.txtOutputLog.append("Normalize: {0}".format(e.normalize))  
				self.txtOutputLog.append("Dropout: {0}, Keep Rate {1} \n".format(e.dropout, e.keep_rate)) 
			elif e.layer_type == 'Max Pool':
				self.txtOutputLog.append('Max Pooling Layer')
				self.txtOutputLog.append("Kernel Size: [1,{0},{1},1], Stride: [1,{2},{3},1]".format(e.kernel_size, e.kernel_size, e.stride, e.stride))
				self.txtOutputLog.append("Padding: {0}".format(e.padding))  
				self.txtOutputLog.append("Normalize: {0}".format(e.normalize))  
				self.txtOutputLog.append("Dropout: {0}, Keep Rate {1} \n".format(e.dropout, e.keep_rate)) 
			elif e.layer_type == 'Fully Connected':
				self.txtOutputLog.append('Fully Connected Layer')
				self.txtOutputLog.append("Num Output Nodes {0}".format(e.num_output_nodes))
				self.txtOutputLog.append("Activation Function: {0}".format(e.act_function))  
				self.txtOutputLog.append("Normalize: {0}".format(e.normalize))  
				self.txtOutputLog.append("Dropout: {0}, Keep Rate {1} \n".format(e.dropout, e.keep_rate)) 
			elif e.layer_type == 'Output':
				self.txtOutputLog.append('Output Layer')
				self.txtOutputLog.append("Activation Function: {0}".format(e.act_function))  
		self.txtOutputLog.append('\n ------------------------------------------------------------------------------------------------------------- \n')

 
	@pyqtSlot(list, list)
	def embed_network_weights_outputs(self, weight_file_names: list, output_file_names: list):

		# clears tabs from previous runs
		self.tabLayerWeights.clear()
		self.tabLayerOutput.clear()

		# cycle 
		for file_name in weight_file_names:
			self.tab = QWidget()
			tab_name = file_name.split('_')
			self.tabLayerWeights.addTab(self.tab, tab_name[0])
			self.image = QLabel()
			self.vbox = QVBoxLayout()
			self.vbox.addWidget(self.image)
			pix_map = QPixmap(file_name)
			self.image.setPixmap(pix_map)
			self.tab.setLayout(self.vbox)

		for file_name in output_file_names:
			self.tab = QWidget()
			tab_name = file_name.split('_')
			self.tabLayerOutput.addTab(self.tab, tab_name[0])
			self.image = QLabel()
			self.vbox = QVBoxLayout()
			self.vbox.addWidget(self.image)
			pix_map = QPixmap(file_name)
			self.image.setPixmap(pix_map)
			self.tab.setLayout(self.vbox)



	@pyqtSlot(bool, int)
	def update_confusion_plot(self, training: bool, epoch: int):
		if training == True:
			plt.savefig('training_confusion_matrix.png', format='png')
			pix_map = QPixmap("training_confusion_matrix.png")
			self.lblTrainingConfusionMat.setPixmap(pix_map)
			self.lblBatchConf.setText("Batch {0}".format(epoch))
		else:
			plt.savefig('testing_confusion_matrix.png', format='png')
			pix_map = QPixmap("testing_confusion_matrix.png")
			self.lblTestingConfusionMat.setPixmap(pix_map)

		plt.close()


	# called when thread(s) have finished, i.e. training has finished or been cancelled
	@pyqtSlot()
	def abort_workers(self):
		for thread, worker in self.__threads: 
			thread.quit()  # this will quit **as soon as thread event loop unblocks**
			thread.wait()  # <- so you need to wait for it to *actually* quit
		self.btnTrainNetwork.setEnabled(True)
	
	def cancel_train(self):
		self.end_train.emit()

		
if __name__ == "__main__":
	app = QApplication(sys.argv)
	ui = CNNApp()
	ui.show()
	sys.exit(app.exec_())
