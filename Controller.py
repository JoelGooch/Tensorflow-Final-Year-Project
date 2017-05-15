# import external dependencies
import sys
import tensorflow as tf
import numpy as np
import pickle
import datetime
import cv2
import xml.etree.ElementTree as ET
import matplotlib as mpl
mpl.use('Qt5Agg')

# import files that I have created myself
import CNN_GUI_V17 as design
import input_data as data
import xml_parser as pp
import Layer as l
import layer_factory as factory
import urllib.error

from functools import partial
from math import ceil, sqrt
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QApplication, QMessageBox, QFileDialog, QSizePolicy, QListWidgetItem, QVBoxLayout
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


class Worker(QObject):

	# signals for updating GUI elements
	epoch_progress = pyqtSignal(float)
	log_message = pyqtSignal(str)
	network_model = pyqtSignal(list)

	# signals for updating/creating visualizations
	batch_loss = pyqtSignal(float, int)
	batch_acc = pyqtSignal(float, int)
	train_valid_loss = pyqtSignal(float, float, int)
	train_valid_acc = pyqtSignal(float, float, int)
	confusion_mat = pyqtSignal(bool, int)
	network_weights = pyqtSignal(list, list)
	network_outputs = pyqtSignal(list, list)

	# signal to inform main thread that worker thread has finished work
	work_complete = pyqtSignal()

	def __init__(self, data_set: str, data_location: str, prima_test_person_out:int, validation: bool, test_split: int, num_epochs: int, batch_size: int, 
		learning_rate: float, momentum: float, optimizer: int, normalize: bool, l2_reg: bool, beta: float, save_directory: str, save_interval: int, update_interval: int, 
		model_path: str, vis_save_path: str, run_time: bool, train_confusion_active: bool, test_confusion_active: bool, conv_weights_active: bool, conv_outputs_active: bool):

		super().__init__()
		self.end_training = False
		self.data_set = data_set
		self.prima_test_person_out = prima_test_person_out
		self.data_location = data_location
		self.regression = False
		self.validation = validation
		self.test_split = test_split
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
		self.update_interval = update_interval
		self.model_path = model_path
		self.vis_save_path = vis_save_path
		self.run_time = run_time
		self.train_confusion_active = train_confusion_active
		self.test_confusion_active = test_confusion_active
		self.conv_weights_active = conv_weights_active
		self.conv_outputs_active = conv_outputs_active
		
	@pyqtSlot()
	def work(self):
		self.train_network(self.data_set, self.data_location, self.prima_test_person_out, self.validation, self.test_split, self.num_epochs, self.batch_size, 
			self.learning_rate, self.momentum, self.optimizer, self.normalize, self.l2_reg, self.beta, self.save_directory, self.save_interval, self.update_interval, 
			self.model_path, self.vis_save_path, self.run_time, self.train_confusion_active, self.test_confusion_active, self.conv_weights_active, self.conv_outputs_active)

	# this function calculates the accuracy measurements from the predicted labels and actual labels
	# 	@param predictions = numpy array containing predicted values by network
	# 	@param labels = numpy array containing actual values  
	def evaluate_accuracy(self, predictions, labels):

		# Convert prediction to degrees
		prediction_degree = (predictions * 180) - 90
		actual_degree = (labels * 180) - 90

		# calculate Root Mean Squared Error 
		RMSE = np.sqrt(np.sum(np.square(prediction_degree - actual_degree), dtype=np.float32) * 1 / predictions.shape[0])
		# calculate Standard Deviation of Root Mean Squared Error 
		RMSE_stdev = np.std(np.sqrt(np.square(prediction_degree - actual_degree)), dtype=np.float32)

		# calculate Mean Absolute Error
		MAE = np.sum(np.absolute(prediction_degree - actual_degree), dtype=np.float32) * 1 / predictions.shape[0]
		# calculate Standard Deviation of Mean Absolute Error
		MAE_stdev = np.std(np.absolute(prediction_degree - actual_degree), dtype=np.float32)

		return RMSE, RMSE_stdev, MAE, MAE_stdev


	def train_network(self, data_set, data_location, prima_test_person_out, validation, test_split, num_epochs, batch_size, learning_rate, momentum, learning_algo, 
		normalize, l2_reg, beta, save_path, save_interval, update_interval, model_path, vis_save_path, run_time, train_confusion_active, test_confusion_active, 
		conv_weights_active, conv_outputs_active):

		try:
			# load selected data set
			if (data_set == 'CIFAR10'):
				training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_CIFAR_10(data_location, validation, test_split)
			if (data_set == 'MNIST'):
				training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_MNIST(data_location, validation, test_split)
			if (data_set == 'PrimaHeadPosePitch'):
				training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_prima_head_pose_pitch(data_location, prima_test_person_out)
				self.regression = True
			if (data_set == 'PrimaHeadPoseYaw'):
				training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_prima_head_pose_yaw(data_location, prima_test_person_out)
				self.regression = True
		except FileNotFoundError:
			self.log_message.emit('File Not Found In Location, Please Select Valid Location in Settings')
			self.work_complete.emit()
			return 0
		except urllib.error.URLError:
			self.log_message.emit('MNIST Not Found In Location, No Internet Connection detected to download data set')
			self.work_complete.emit()
			return 0


		# normalize data, if user requires
		if normalize == True:
			training_set -= 127 
			testing_set -= 127
			if validation == True:
				validation_set -= 127

		try:
			# get array of layers from XML file
			layers = pp.get_layers(model_path)
		except FileNotFoundError:
			self.log_message.emit('Error Reading XML File. Could Not Locate XML File')
			self.work_complete.emit()
			return 0
		except ET.ParseError:
			self.log_message.emit('Error Reading XML File. XML File is badly formed')
			self.work_complete.emit()
			return 0

		self.network_model.emit(layers)

		self.log_message.emit('Initialising Tensorflow Variables... \n')
		
		graph = tf.Graph()
		with graph.as_default():

			# define placeholder variables
			x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
			y = tf.placeholder(tf.float32, shape=(None, num_classes))

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

				try:
					# create as many layers as are stored in XML file
					for e in range(len(layers)):
						layer, input_dimension = layer_factory.create_layer(layer, layers[e], input_dimension)
						network_layers.append(layer)

				except:
					self.log_message.emit('Error Reading XML File.')

				# return last element of layers array (output layer)
				return network_layers[-1]


			model_output = CNN_Model(x)

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


			# intitialise Tensorflow optiimiser in computational graph, dependent upon previous user selection
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

				# start recording time taken to train
				train_start = datetime.datetime.now()

				# start training loop
				for epoch in range(num_epochs + 1):
					# check if user has signalled training to be cancelled
					if self.end_training == False:
						# grab random batch from training data
						offset = (epoch * batch_size) % (training_labels.shape[0] - batch_size)
						batch_data = training_set[offset:(offset + batch_size), :, :, :]
						batch_labels = training_labels[offset:(offset + batch_size)]

						# feed batch through network     
						feed_dict = {x: batch_data, y: batch_labels}


						if self.regression == False:
							_, batch_loss, batch_accuracy, batch_pred_class, batch_classes = session.run([optimizer, loss, accuracy, network_pred_class, class_labels], feed_dict=feed_dict)
						else: 
							_, batch_loss, batch_pred_angles = session.run([optimizer, loss, model_output], feed_dict=feed_dict)

						# print information to output log
						if (epoch % self.update_interval == 0):
							self.log_message.emit('')
							self.log_message.emit('Loss at epoch: {} of {} is {}'.format(epoch, str(num_epochs), batch_loss))
							self.log_message.emit('Global Step: {}'.format(str(global_step.eval())))

							if self.regression == False:
								self.log_message.emit('Batch Accuracy = {}%'.format(str(batch_accuracy * 100)))
								self.batch_acc.emit((batch_accuracy * 100), epoch)
							else: 
								RMSE, RMSE_stdev, MAE, MAE_stdev = self.evaluate_accuracy(batch_pred_angles, batch_labels)
								self.log_message.emit('Batch Root Mean Squared Error = {}'.format(str(RMSE)))
								self.log_message.emit('Batch Root Mean Squared Error Standard Deviation = {}'.format(str(RMSE_stdev)))
								self.log_message.emit('Batch Mean Absolute Error = {}'.format(str(MAE)))
								self.log_message.emit('Batch Mean Absolute Error Standard Deviation = {}'.format(str(MAE_stdev)))
								self.batch_acc.emit(RMSE, epoch)
							
							# emit batch loss for GUI visualization
							self.batch_loss.emit(batch_loss, epoch)
							

							# if user has chosen to include run time visualizations
							if run_time == True:

								if self.regression == False:
									if train_confusion_active == True:
										# create confusion matrix for current batch
										batch_confusion = tf.contrib.metrics.confusion_matrix(batch_classes, batch_pred_class).eval()
										self.create_confusion_matrix(batch_confusion, vis_save_path, True, epoch)
										self.confusion_mat.emit(True, epoch)

								if validation == True:

									num_training_data = len(training_labels)
									# this value needs to be decided and likely smaller, will mean machine requirements to run program are high
									data_per_batch = 5000
									num_batches = num_training_data / data_per_batch

									train_accuracy_total = 0
									train_loss_total = 0

									for i in range(int(num_batches)):
									   batch_data = training_set[i*data_per_batch:(i+1)*data_per_batch]
									   batch_labels = training_labels[i*data_per_batch:(i+1)*data_per_batch]
									   train_accuracy, train_loss = session.run([accuracy, loss], feed_dict={x: batch_data, y: batch_labels})
									   train_accuracy_total += train_accuracy
									   train_loss_total += train_loss
									  
									train_accuracy_total /= num_batches
									train_loss_total /= num_batches

									# run validation set at current batch
									valid_accuracy, valid_loss = session.run([accuracy, loss], feed_dict={x: validation_set, y: validation_labels})
									self.train_valid_acc.emit((train_accuracy * 100), (valid_accuracy * 100), epoch)
									self.train_valid_loss.emit(train_loss_total, valid_loss, epoch)


						# calculate progress as percentage and emit signal to GUI to update
						epoch_prog = (epoch / num_epochs) * 100
						self.epoch_progress.emit(epoch_prog)

						# check save interval to avoid divide by zero
						if save_interval != 0:
							try:
								# save at interval define by user
								if (epoch % save_interval == 0):
									saver.save(sess=session, save_path=save_path, global_step=global_step)
									self.log_message.emit('Saved Checkpoint \n')
							except ValueError:
								self.log_message.emit('Directory to save checkpoints to does not exist. Please Change. Checkpoint NOT Saved.')

				self.log_message.emit('\nTraining Complete')
				train_end = datetime.datetime.now()
				training_time = train_end - train_start
				self.log_message.emit(('Training Took ' + str(training_time)) + '\n')

				try:
					# save network state when complete
					saver.save(sess=session, save_path=save_path, global_step=global_step)
					self.log_message.emit('Saved Checkpoint')
				except ValueError:
					self.log_message.emit('Directory to save checkpoints to does not exist. Please Change. Checkpoint NOT Saved.')

				# set progress bar to 100% (if training was not interrupted)
				if (self.end_training == False):
					self.epoch_progress.emit(100)

				self.log_message.emit('\nEvaluating Test Set...')

				feed_dict = {x: testing_set, y: testing_labels}

				# run test set through network
				if self.regression == False:
					# if classification, grab accuracy from tensorflow graph
					test_acc, testing_pred_class, testing_classes = session.run([accuracy, network_pred_class, class_labels], feed_dict=feed_dict)
					self.log_message.emit('Test Set Accuracy = {}%'.format(str(test_acc * 100)))
				else: 
					# otherwise calculate RMSE using function
					batch_pred_angles = session.run(model_output, feed_dict=feed_dict)
					RMSE, RMSE_stdev, MAE, MAE_stdev = self.evaluate_accuracy(batch_pred_angles, testing_labels)
					self.log_message.emit('Test Set Root Mean Squared Error = {}'.format(str(RMSE)))
					self.log_message.emit('Test Set Root Mean Squared Error Standard Deviation = {}'.format(str(RMSE_stdev)))
					self.log_message.emit('Test Set Mean Absolute Error = {}'.format(str(MAE)))
					self.log_message.emit('Test Set Mean Absolute Error Standard Deviation = {}'.format(str(MAE_stdev)))

				self.log_message.emit('Test Set Evaluated \n')
				self.log_message.emit('Loading Visualizations...')


				

				conv_layer_names = []

				for layer in layers:
					if layer.layer_type == 'Convolution':
						conv_layer_names.append(layer.layer_name)

				# generate visualization of convolution layer weights (if user requested)
				if conv_weights_active == True:
					layer_weights_file_names = []
	
					for e in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
						if 'Cweights' in e.name:
							file_name = self.plot_conv_weights(weights=e, vis_save_path=vis_save_path, name=e.name, session=session)
							layer_weights_file_names.append(file_name)

					self.network_weights.emit(layer_weights_file_names, conv_layer_names)

				# generate visualization of convolution layer outputs (if user requested)
				if conv_outputs_active == True:
					layer_outputs_file_names = []
					
					# grab random image from test set
					random = np.random.randint(0, testing_set.shape[0])
					image = testing_set[random]
					layer_count = 0
					# create convolution output plots for each convolution layer
					for layer in network_layers:
						if layers[layer_count].layer_type == 'Convolution':
							file_name = self.plot_conv_layer(layer=layer, vis_save_path=vis_save_path, name=layers[layer_count].layer_name, image=image, session=session, x=x)
							layer_outputs_file_names.append(file_name)
						layer_count += 1

					self.network_outputs.emit(layer_outputs_file_names, conv_layer_names)	

				# generate test set confusion matrix
				if self.regression == False:

					if test_confusion_active == True:
						# create confusion matrix from predicted and actual classes
						test_set_confusion = tf.contrib.metrics.confusion_matrix(testing_classes, testing_pred_class).eval()
						self.create_confusion_matrix(test_set_confusion, vis_save_path, False, 0)
						self.confusion_mat.emit(False, 0)
							

				self.log_message.emit('Visualizations Loaded\n')

				# signal that thread has completed
				self.work_complete.emit()


	def cancel_training(self):
		self.end_training = True

	def plot_conv_layer(self, layer, vis_save_path, name, image, session, x):
		try:
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

			file_name = vis_save_path + name + "_output.png"
			plt.savefig(file_name, format='png')
			plt.close()
			return file_name
		except FileNotFoundError:
			self.log_message.emit('Directory chosen to save visualizations was not found. Please amend in Settings.')

	def plot_conv_weights(self, weights, vis_save_path, name, session, input_channel=0):
		try:
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
			file_name = vis_save_path + name + ".png"
			plt.savefig(file_name, format='png')
			plt.close()
			return file_name

		except FileNotFoundError:
			return 0

	def create_confusion_matrix(self, confusion, vis_save_path, training, batch):

		norm_conf = []

		try:
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
				plt.savefig(vis_save_path + 'training_confusion_matrix_batch_' + str(batch) + '.png', format='png')
			else:
				plt.savefig(vis_save_path + 'testing_confusion_matrix.png', format='png')
		# sometimes get a division by zero error, this just skips creating that matrix
		except ZeroDivisionError:
			return 0
		except FileNotFoundError:
			return 0


# this is a simple custom canvas that i can embed into a pyqt5 widget.
class MyMplCanvas(FigureCanvas):

	def __init__(self, parent=None, width=20, height=20, dpi=100, xAxisTitle='x', yAxisTitle='y'):
		self.xAxisTitle = xAxisTitle
		self.yAxisTitle = yAxisTitle

		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = self.fig.add_subplot(111)

		self.axes.set_xlabel(self.xAxisTitle)
		self.axes.set_ylabel(self.yAxisTitle)

		self.axes.grid(True)

		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)

		FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)

# this graph gets updated within the application
class DynamicGraph(MyMplCanvas):
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
		
		self.draw()

# this graph is is the same as above except with two lines, for validation/training accuracy
class DynamicDoubleGraph(MyMplCanvas):
	def __init__(self, *args, **kwargs):
		MyMplCanvas.__init__(self, *args, **kwargs)
		self.train_errors = []
		self.valid_errors = []
		self.epochs = []
		self.legend = self.axes.legend(loc='upper center', shadow=True)

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
		self.spnConvKeepRate.setEnabled(False)
		self.spnPoolKeepRate.setEnabled(False)
		self.spnFCKeepRate.setEnabled(False)
		self.btnCreateModel.setEnabled(False)
		self.spnRegBeta.setEnabled(False)

		# for quickness. TEMPORARY
		self.txtLoadModel.setText('C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/Tensorflow-Final-Year-Project/Models/MNIST_Tensorflow_Official.xml')

		# navigational buttons
		self.actionTrain.triggered.connect(partial(self.open_tab, index=0))
		self.actionDesign.triggered.connect(partial(self.open_tab, index=1))
		self.actionVisualizations.triggered.connect(partial(self.open_tab, index=2))
		self.actionSettings.triggered.connect(partial(self.open_tab, index=3))
		self.actionExit.triggered.connect(self.close)

		self.cbxSavePath.stateChanged.connect(self.checkpoint_checkbox_state_changed)
		self.radCreateModel.toggled.connect(self.create_model_rad_clicked)
		self.spnTestSplit.valueChanged.connect(self.testing_validation_split_spinner_changed)

		# events for checking/unchecking dropout when creating model
		self.chkConvDropout.stateChanged.connect(partial(self.dropout_checkbox_state_changed, dropout_checkbox=self.chkConvDropout, keep_rate_spinner=self.spnConvKeepRate))
		self.chkPoolDropout.stateChanged.connect(partial(self.dropout_checkbox_state_changed, dropout_checkbox=self.chkPoolDropout, keep_rate_spinner=self.spnPoolKeepRate))
		self.chkFCDropout.stateChanged.connect(partial(self.dropout_checkbox_state_changed, dropout_checkbox=self.chkFCDropout, keep_rate_spinner=self.spnFCKeepRate))

		self.cbxOptimizer.currentIndexChanged.connect(self.optimizer_combo_box_changed)
		self.chkL2Reg.stateChanged.connect(self.l2_reg_checkbox_state_changed)

		self.cbxConvBiasInit.currentIndexChanged.connect(partial(self.bias_init_combo_changed, bias_init_combobox=self.cbxConvBiasInit, 
			bias_val_spinner= self.spnConvBiasVal, bias_val_label= self.lblConvBiasVal, help_icon=self.hlpConvBiasVal))
		self.cbxFCBiasInit.currentIndexChanged.connect(partial(self.bias_init_combo_changed, bias_init_combobox=self.cbxFCBiasInit, 
			bias_val_spinner= self.spnFCBiasVal, bias_val_label= self.lblFCBiasVal, help_icon=self.hlpFCBiasVal))
		self.cbxOutputBiasInit.currentIndexChanged.connect(partial(self.bias_init_combo_changed, bias_init_combobox=self.cbxOutputBiasInit, 
			bias_val_spinner= self.spnOutputBiasVal, bias_val_label= self.lblOutputBiasVal, help_icon=self.hlpOutputBiasVal))

		# buttons to change file paths 
		self.btnChangeSavePath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtSavePath))
		self.btnChangeLoadCheckpoints.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtLoadCheckpoints, disable=True))
		self.btnChangeModelPath.clicked.connect(self.change_file_path)
		self.btnChangeModelSavePath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtModelSavePath))
		self.btnChangeMNISTPath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtMNISTPath))
		self.btnChangeCIFARPath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtCIFARPath))
		self.btnChangePrimaPitchPath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtPrimaPitchPath))
		self.btnChangePrimaYawPath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtPrimaYawPath))

		# train/cancel training buttons
		self.btnTrainNetwork.clicked.connect(self.train_button_clicked)
		self.btnCancelTraining.clicked.connect(self.cancel_train)

		# connect events for when data set selection radio buttons are toggled
		self.radMNIST.toggled.connect(partial(self.data_set_rad_state_changed, False))
		self.radCIFAR10.toggled.connect(partial(self.data_set_rad_state_changed, False))
		self.radPrimaHeadPosePitch.toggled.connect(partial(self.data_set_rad_state_changed, True))
		self.radPrimaHeadPoseYaw.toggled.connect(partial(self.data_set_rad_state_changed, True))

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
		self.batch_loss_graph = DynamicGraph(self.grphBatchLoss, width=5, height=4, dpi=100, xAxisTitle='Epoch', yAxisTitle='Loss')
		self.batch_acc_graph = DynamicGraph(self.grphBatchAcc, width=5, height=4, dpi=100, xAxisTitle='Epoch', yAxisTitle='Accuracy (%)')
		self.train_valid_accuracy = DynamicDoubleGraph(self.grphTrainValidAcc, width=5, height=4, dpi=100, xAxisTitle='Epoch', yAxisTitle='Accuracy (%)')
		self.train_valid_loss = DynamicDoubleGraph(self.grphTrainValidLoss, width=5, height=4, dpi=100, xAxisTitle='Epoch', yAxisTitle='Loss')

		
	def train_button_clicked(self):
		# clear visualizations of previous runs
		self.reset_visualizations()

		try:
			num_epochs = int(self.spnNumEpochs.text())
			batch_size = int(self.spnBatchSize.text())
			learning_rate = float(self.spnLearningRate.text())
			optimizer = int(self.cbxOptimizer.currentIndex())

			if self.current_data_set() == 'CIFAR10':
				data_location = self.txtCIFARPath.text()
			elif self.current_data_set() == 'MNIST':
				data_location = self.txtMNISTPath.text()
			elif self.current_data_set() == 'PrimaHeadPosePitch':
				data_location = self.txtPrimaPitchPath.text()
			elif self.current_data_set() == 'PrimaHeadPoseYaw':
				data_location = self.txtPrimaYawPath.text()
			
			
			# obtain model file path and checkpoint save path from user text fields
			model_path = self.txtLoadModel.text()
			save_path = self.txtSavePath.text()

			prima_test_person_out = self.cbxPrimaTestPerson.currentIndex() + 1

			if self.chkValidationSet.isChecked():
				validation = True
				test_split = int(self.spnTestSplit.text())
			else: 
				validation = False
				test_split = 0

			if self.cbxOptimizer.currentIndex == 4:
				momentum = float(self.spnMomentum.text())
			else: momentum = 0

			if self.chkNormalize.isChecked():
				normalize = True
			else: normalize = False

			if self.chkL2Reg.isChecked():
				l2_reg = True
				beta = float(self.spnRegBeta.text())
			else:
				l2_reg = False
				beta = 0

			if self.chkVisualizations.isChecked():
				run_time = True
			else: run_time = False

			if self.cbxSaveInterval.currentIndex() == 0:
				save_interval = 0
			else: save_interval = int(self.cbxSaveInterval.currentText())

			update_interval = int(self.spnUpdateInterval.text())

			vis_save_path = self.txtVisSavePath.text()

			if self.chkTrainConf.isChecked():
				train_confusion_active = True
			else: train_confusion_active = False

			if self.chkTestConf.isChecked():
				test_confusion_active = True
			else: test_confusion_active = False

			if self.chkConvWeights.isChecked():
				conv_weights_active = True
			else: conv_weights_active = False

			if self.chkConvOutputs.isChecked():
				conv_outputs_active = True
			else: conv_outputs_active = False
	
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
			worker = Worker(self.current_data_set(), data_location, prima_test_person_out, validation, test_split, num_epochs, batch_size, learning_rate, 
				momentum, optimizer, normalize, l2_reg, beta, save_path, save_interval, update_interval, model_path, vis_save_path, run_time, train_confusion_active, 
				test_confusion_active, conv_weights_active, conv_outputs_active)

			thread = QThread()

			# connect cancel button in main thread to background thread
			self.end_train.connect(worker.cancel_training) # THIS WOULD NOT WORK A FEW LINES LOWER!!!!

			# store reference to objects so they are not garbage collected
			self.__threads.append((thread, worker))
			worker.moveToThread(thread)

			# set connections from background thread to main thread for updating GUI elements
			worker.epoch_progress.connect(self.update_progress_bar)
			worker.log_message.connect(self.txtOutputLog.append)
			worker.network_model.connect(self.show_model_details)
			worker.work_complete.connect(self.abort_workers)

			# set connections from background thread to create/update visualizations on GUI
			worker.batch_loss.connect(self.batch_loss_graph.update_figure)
			worker.batch_acc.connect(self.batch_acc_graph.update_figure)
			worker.train_valid_loss.connect(self.train_valid_loss.update_figure)
			worker.train_valid_acc.connect(self.train_valid_accuracy.update_figure)
			worker.confusion_mat.connect(self.update_confusion_plot)
			worker.network_weights.connect(self.embed_network_weights)
			worker.network_outputs.connect(self.embed_network_outputs)

			# connect and start thread
			thread.started.connect(worker.work)
			thread.start()

	def create_conv_layer_button_clicked(self):
		if not self.txtConvName.text():
				self.txtOutputModelLog.append('Layer Must have a Name!')
				return 0
		try:
			conv_layer_name = self.txtConvName.text()

			if self.cbxConvKernelSize.currentIndex() == 0:
				conv_kernel_size = 3
			elif self.cbxConvKernelSize.currentIndex() == 1:
				conv_kernel_size = 5
			elif self.cbxConvKernelSize.currentIndex() == 2:
				conv_kernel_size = 7

			conv_stride = self.return_stride(self.cbxConvStride)
			num_output_filters = int(self.spnConvOutputFilters.text())
			act_function = self.return_act_function(self.cbxConvActFunction)
			weight_init, weight_std_dev = self.return_weight_init(self.cbxConvWeightInit, self.spnConvStdDev)
			bias_init, bias_val = self.return_bias_init(self.cbxConvBiasInit, self.spnConvBiasVal)
			padding = self.return_padding(self.cbxConvPadding)
			normalize = self.return_normalize(self.chkConvNorm)
			dropout, keep_rate = self.return_dropout(self.chkConvDropout, self.spnConvKeepRate)

			layer = l.ConvLayer(conv_layer_name, conv_kernel_size, conv_stride, act_function, num_output_filters, weight_init, weight_std_dev, bias_init, bias_val, padding, normalize, dropout, keep_rate)
			self.new_model.append(layer)

			item = QListWidgetItem(("Convolution, Num Output Filters {}, Kernel Size: {}, Stride: {}, Activation Function: {}, Padding: {}, Normalize: {}, Dropout: {}, Keep Rate: {}").format(layer.num_output_filters, layer.kernel_size, layer.stride, layer.act_function, layer.padding, layer.normalize, layer.dropout, layer.keep_rate))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	def create_pooling_layer_button_clicked(self):
		if not self.txtPoolName.text():
			self.txtOutputModelLog.append('Layer Must have a Name!')
			return 0
		try:    
			pool_layer_name = self.txtPoolName.text()

			if self.cbxPoolKernelSize.currentIndex() == 0:
				pool_kernel_size = 2
			elif self.cbxPoolKernelSize.currentIndex() == 1:
				pool_kernel_size = 3
			elif self.cbxPoolKernelSize.currentIndex() == 2:
				pool_kernel_size = 4
			elif self.cbxPoolKernelSize.currentIndex() == 3:
				pool_kernel_size = 5

			pool_stride = self.return_stride(self.cbxPoolStride)
			padding = self.return_padding(self.cbxPoolPadding)
			normalize = self.return_normalize(self.chkPoolNorm)
			dropout, keep_rate = self.return_dropout(self.chkPoolDropout, self.spnPoolKeepRate)

			layer = l.MaxPoolingLayer(pool_layer_name, pool_kernel_size, pool_stride, padding, normalize, dropout, keep_rate)
			self.new_model.append(layer)

			item = QListWidgetItem(("Max Pool, Kernel Size: {}, Stride: {}, Padding: {}, Normalize: {}, Dropout: {}, Keep Rate: {}").format(layer.kernel_size, layer.stride, layer.padding, layer.normalize, layer.dropout, layer.keep_rate))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	def create_full_conn_layer_button_clicked(self):
		if not self.txtFCName.text():
			self.txtOutputModelLog.append('Layer Must have a Name!')
			return 0
		try:
			FC_layer_name = self.txtFCName.text()
			num_output_nodes = int(self.spnFCNumOutputNodes.text())
			act_function = self.return_act_function(self.cbxFCActFunction)
			weight_init, weight_std_dev = self.return_weight_init(self.cbxFCWeightInit, self.spnFCStdDev)
			bias_init, bias_val = self.return_bias_init(self.cbxFCBiasInit, self.spnFCBiasVal)
			dropout, keep_rate = self.return_dropout(self.chkFCDropout, self.spnFCKeepRate)

			layer = l.FullyConnectedLayer(FC_layer_name, act_function, num_output_nodes, weight_init, weight_std_dev, bias_init, bias_val, dropout, keep_rate)
			self.new_model.append(layer)

			item = QListWidgetItem(("Fully Connected,  Num Output Nodes: {}, Activation Function: {}, Dropout: {}, Keep Rate: {}").format(layer.num_output_nodes, layer.act_function, layer.dropout, layer.keep_rate))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	def create_output_layer_button_clicked(self):
		if not self.txtOutputName.text():
			self.txtOutputModelLog.append('Layer Must have a Name!')
			return 0	
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

			weight_init, weight_std_dev = self.return_weight_init(self.cbxOutputWeightInit, self.spnOutputStdDev)
			bias_init, bias_val = self.return_bias_init(self.cbxOutputBiasInit, self.spnOutputBiasVal)

			layer = l.OutputLayer(output_layer_name, act_function, weight_init, weight_std_dev, bias_init, bias_val)
			self.new_model.append(layer)

			item = QListWidgetItem(("Output, Activation Function: {}").format(layer.act_function))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	def return_stride(self, comboBox):
		if comboBox.currentIndex() == 0:
			stride = 1
		elif comboBox.currentIndex() == 1:
			stride = 2
		elif comboBox.currentIndex() == 2:
			stride = 3
		return stride

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
			file_name = self.txtSaveModelAs.text()
			if not file_name:
				self.txtOutputModelLog.append("Please add name for new model")
				return False

			file_path = self.txtModelSavePath.text()

			success = pp.create_XML_model(self.new_model, file_name, file_path)

			if success == True:
				self.txtOutputModelLog.append("Success Writing XML File")
				response = QMessageBox.question(self, "Model Creation Successful", "Would you like to use the newly created model?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
				if response == QMessageBox.Yes:
					self.txtLoadModel.setText((file_path + file_name + '.xml'))
					self.tabPages.setCurrentIndex(0)
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
		self.cbxConvKernelSize.setCurrentIndex(0)
		self.cbxConvStride.setCurrentIndex(0)
		self.cbxConvActFunction.setCurrentIndex(0)
		self.spnConvOutputFilters.setValue(64)
		self.cbxConvWeightInit.setCurrentIndex(0)
		self.cbxConvBiasInit.setCurrentIndex(0)
		self.spnConvStdDev.setValue(0.005)
		self.spnConvBiasVal.setValue(0.005)
		self.cbxConvPadding.setCurrentIndex(0)
		self.chkConvNorm.setChecked(False)
		self.chkConvDropout.setChecked(False)
		self.spnConvKeepRate.setValue(0.5)
		self.txtPoolName.setText('')
		self.cbxPoolKernelSize.setCurrentIndex(0)
		self.cbxPoolStride.setCurrentIndex(0)
		self.cbxPoolPadding.setCurrentIndex(0)
		self.chkPoolNorm.setChecked(False)
		self.chkPoolDropout.setChecked(False)
		self.spnPoolKeepRate.setValue(0.5)
		self.txtFCName.setText('')
		self.cbxFCActFunction.setCurrentIndex(0)
		self.spnFCNumOutputNodes.setValue(64)
		self.cbxFCWeightInit.setValue(0.005)
		self.cbxFCBiasInit.setValue(0.005)
		self.spnFCStdDev.setValue(0.005)
		self.spnFCBiasVal.setValue(0.005)
		self.chkFCNorm.setChecked(False)
		self.chkFCDropout.setChecked(False)
		self.spnFCKeepRate.setValue(0.5)
		self.txtOutputName.setText('')
		self.cbxOutputActFunction.setCurrentIndex(0)
		self.cbxOutputWeightInit.setCurrentIndex(0)
		self.spnOutputBiasVal.setValue(0.005)
		self.spnOutputStdDev.setValue(0.005)
		self.spnOutputBiasVal.setValue(0.005)

	def reset_visualizations(self):
		self.lblTrainingConfusionMat.setText('Training Confusion Matrix for First Batch Will Appear When it has been Evaluated')
		self.lblTestingConfusionMat.setText('Test Set Confusion Matrix Will Appear When Test Set has been Evaluated')
		self.tabLayerWeights.clear()
		self.tabLayerOutput.clear()

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

	def bias_init_combo_changed(self, bias_init_combobox, bias_val_spinner, bias_val_label, help_icon):
		if bias_init_combobox.currentIndex() == 0:
			bias_val_label.setText('')
			bias_val_spinner.setFixedWidth(0)
			help_icon.setFixedWidth(0)
		elif bias_init_combobox.currentIndex() == 1 or bias_init_combobox.currentIndex() == 2:
			bias_val_label.setText('Std Dev of Weights:')
			bias_val_spinner.setFixedWidth(60)
			help_icon.setFixedWidth(24)
		elif bias_init_combobox.currentIndex() == 3:
			bias_val_label.setText('Value:')
			bias_val_spinner.setFixedWidth(60)
			help_icon.setFixedWidth(24)

	def testing_validation_split_spinner_changed(self):
		test_split = self.spnTestSplit.text()
		valid_split = 100 - int(test_split)
		self.txtValidSplit.setText(str(valid_split))

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
			self.cbxPrimaTestPerson.setFixedWidth(41)
			self.lblPrimaTest.setText('Use Person')
			self.lblPrimaTest2.setText('as Testing Set')
		else: 
			self.chkValidationSet.setFixedWidth(110)
			self.cbxPrimaTestPerson.setFixedWidth(0)
			self.lblPrimaTest.setText('')
			self.lblPrimaTest2.setText('')

	def current_data_set(self):
		if self.radCIFAR10.isChecked():
			return 'CIFAR10'
		elif self.radMNIST.isChecked():
			return 'MNIST'
		elif self.radPrimaHeadPosePitch.isChecked():
			return 'PrimaHeadPosePitch'
		elif self.radPrimaHeadPoseYaw.isChecked():
			return 'PrimaHeadPoseYaw'

	def optimizer_combo_box_changed(self):
		if self.cbxOptimizer.currentIndex() == 4:
			self.spnMomentum.setFixedWidth(50)
			self.spnMomentum.setFixedHeight(20)
			self.lblMomentum.setText('Momentum:')
		else:
			self.spnMomentum.setFixedWidth(0)
			self.lblMomentum.setText('')   

	def l2_reg_checkbox_state_changed(self):
		if self.chkL2Reg.isChecked():
			self.spnRegBeta.setEnabled(True)
		else: self.spnRegBeta.setEnabled(False)

	def checkpoint_checkbox_state_changed(self):
		if self.cbxSavePath.isChecked():
			self.txtSavePath.setText(self.txtLoadCheckpoints.text())

	def dropout_checkbox_state_changed(self, dropout_checkbox, keep_rate_spinner):
		if dropout_checkbox.isChecked():
			keep_rate_spinner.setEnabled(True)
		else: keep_rate_spinner.setEnabled(False)

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
				self.txtOutputLog.append("Dropout: {0}, Keep Rate {1} \n".format(e.dropout, e.keep_rate)) 
			elif e.layer_type == 'Output':
				self.txtOutputLog.append('Output Layer')
				self.txtOutputLog.append("Activation Function: {0}".format(e.act_function))  
		self.txtOutputLog.append('\n------------------------------------------------------------------------------------------------------------- \n')

 
	@pyqtSlot(list, list)
	def embed_network_weights(self, weight_file_names: list, conv_layer_names: list):
		try:
			count = 0
			for file_name in weight_file_names:
				self.tab = QWidget()
				tab_name = conv_layer_names[count]
				self.tabLayerWeights.addTab(self.tab, tab_name)
				self.image = QLabel()
				self.vbox = QVBoxLayout()
				self.vbox.addWidget(self.image)
				pix_map = QPixmap(file_name)
				self.image.setPixmap(pix_map)
				self.tab.setLayout(self.vbox)
				count += 1
		except AttributeError:
			return 0

	@pyqtSlot(list, list)
	def embed_network_outputs(self, output_file_names: list, conv_layer_names: list):
		try:
			count = 0
			for file_name in output_file_names:
				self.tab = QWidget()
				tab_name = conv_layer_names[count]
				self.tabLayerOutput.addTab(self.tab, tab_name)
				self.image = QLabel()
				self.vbox = QVBoxLayout()
				self.vbox.addWidget(self.image)
				pix_map = QPixmap(file_name)
				self.image.setPixmap(pix_map)
				self.tab.setLayout(self.vbox)	
				count += 1
		except AttributeError:
			return 0

	# this function loads the created confusion matrix from disk and loads into GUI
	@pyqtSlot(bool, int)
	def update_confusion_plot(self, training: bool, epoch: int):
		# boolean path way that states whether or not this confusion matrix is for training or testing
		if training == True:
			plt.savefig('training_confusion_matrix.png', format='png')
			pix_map = QPixmap("training_confusion_matrix.png")
			self.lblTrainingConfusionMat.setPixmap(pix_map)
			self.lblBatchConf.setText("Training Batch {0}".format(epoch))
			plt.close()
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
	
	# function that signals to the worked thread that the user has requested training to stop
	def cancel_train(self):
		self.end_train.emit()

		
if __name__ == "__main__":
	app = QApplication(sys.argv)
	ui = CNNApp()
	ui.show()
	sys.exit(app.exec_())
