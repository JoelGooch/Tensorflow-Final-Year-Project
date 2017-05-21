# import external dependencies
import sys
import tensorflow as tf
import numpy as np
import pickle
import datetime
import xml.etree.ElementTree as ET
import matplotlib as mpl
mpl.use('Qt5Agg')

# import files that I have created myself
import CNN_GUI_V19 as design
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


# this is the background thread class
class Worker(QObject):

	# signals for updating GUI elements
	epoch_progress = pyqtSignal(float)  # signal that updates the progress bar
	log_message = pyqtSignal(str)		# signal that appends text to the Train window output log
	network_model = pyqtSignal(list)	# signal that emits the architecture read in from .XML and writes to log

	# signals for updating/creating visualizations
	batch_loss = pyqtSignal(float, int)					# for updating the batch loss over epochs graph
	batch_acc = pyqtSignal(float, int)					# for updating the batch accuracy over epochs graph
	train_valid_loss = pyqtSignal(float, float, int)	# for updating the training/validation loss graph
	train_valid_acc = pyqtSignal(float, float, int)		# for updating the training/validation accuracy graph
	confusion_mat = pyqtSignal(str, int, bool)			# for updating the training/testing confusion matrix visualization
	network_weights = pyqtSignal(list, list)			# for updating the convolution layer weights visualization
	network_outputs = pyqtSignal(list, list)			# for updating the convolution layer outputs visualization
	embed_sample_images = pyqtSignal(str, bool)			# for updating the sample/predicted images visualization
	show_classification_key = pyqtSignal(bool)			# shows the cifar-10 or prima confusion matrix key

	change_graph_titles = pyqtSignal(bool)					# for changing the main titles on the validation/training accuracy/loss graphs when switching between regression and classification tasks
	change_accuracy_graph_axis = pyqtSignal(bool, str)		# for changing the x axis title on the validation/training accuracy graph when switching between regression and classification tasks
	change_loss_graph_axis = pyqtSignal(bool, str)			# for changing the x axis title on the validation/training loss graph when switching between regression and classification tasks

	work_complete = pyqtSignal()	# signal to inform main thread that worker thread has finished work

	def __init__(self, data_set: str, data_location: str, prima_test_person_out:int, validation: bool, test_split: int, num_epochs: int, batch_size: int, 
		learning_rate: float, momentum: float, optimizer: int, normalize: bool, l2_reg: bool, beta: float, save_directory: str, save_interval: int, update_interval: int, 
		model_path: str, vis_save_path: str, run_time: bool, train_confusion_active: bool, test_confusion_active: bool, conv_weights_active: bool, conv_outputs_active: bool,
		save_to_csv: bool, csv_save_path: str):

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
		self.save_to_csv = save_to_csv
		self.csv_save_path = csv_save_path
		
	# called to commence Tensorflow training in the background
	@pyqtSlot()
	def work(self):
		try:
			self.train_network(self.data_set, self.data_location, self.prima_test_person_out, self.validation, self.test_split, self.num_epochs, self.batch_size, 
				self.learning_rate, self.momentum, self.optimizer, self.normalize, self.l2_reg, self.beta, self.save_directory, self.save_interval, self.update_interval, 
				self.model_path, self.vis_save_path, self.run_time, self.train_confusion_active, self.test_confusion_active, self.conv_weights_active, self.conv_outputs_active,
				self.save_to_csv, self.csv_save_path)
		except tf.errors.ResourceExhaustedError: 
			self.log_message.emit('\nResource Exhausted Error, Your GPU does not have enough memory to train this network\n')
			self.work_complete.emit()
		except tf.errors.InvalidArgumentError:
			self.log_message.emit('\nAn Unexpected Error has Occured. Please check the .XML file.\n')
			self.work_complete.emit()
		except ValueError:
			self.log_message.emit('\nAn Unexpected Error has Occured. Please check the .XML file.\n')
			self.work_complete.emit()
		except IndexError:
			self.log_message.emit('\nAn Unexpected Error has Occured. Please check the .XML file.\n')
			self.work_complete.emit()


	# this function calculates the accuracy measurements from the predicted labels and actual labels
	# 	@param predictions = numpy array containing predicted values by network
	# 	@param labels = numpy array containing actual values  
	# 	@param write_to_csv = boolean that states whether to write performance details to .csv
	# 	@param file_path = string that contains diretory to write .csv details to
	def evaluate_accuracy(self, predictions, labels, write_to_csv, file_path):

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

		# if user wants results to be written to .csv file
		try:
			if write_to_csv == True:

				num_rows, _ = labels.shape
				now = datetime.datetime.now()

				# open .csv file and write headings for accuracy measurements
				file = open(file_path, 'a')
				file.write('\nTesting Set Performance \nRMSE = ' + str(RMSE) + ',RMSE Standard Deviation = ' + str(RMSE_stdev) + ',MAE = ' + str(MAE) + ',MAE Standard Deviation = ' + str(MAE_stdev) + '\n')
				file.write('\nPredicted Degree,Actual Degree,Error \n')

				# write predicted, actual and error between predictions for each label in the test set
				for i in range(0, num_rows):
					error = (prediction_degree[i] - actual_degree[i])
					file.write(str(prediction_degree[i]) + ',' + str(actual_degree[i]) + ',' + str(error) + '\n')
				
				file.close()

		# skip this if the user has opened the file when it is being written, or changed persmissions on file
		except PermissionError:
			self.log_message.emit('Permission Denied opening log file, please close the file or change the permissions')

		return RMSE, RMSE_stdev, MAE, MAE_stdev


	# commences training and testing of network in Tensorflow in background thread
	def train_network(self, data_set, data_location, prima_test_person_out, validation, test_split, num_epochs, batch_size, learning_rate, momentum, learning_algo, 
		normalize, l2_reg, beta, save_path, save_interval, update_interval, model_path, vis_save_path, run_time, train_confusion_active, test_confusion_active, 
		conv_weights_active, conv_outputs_active, write_to_csv, csv_save_path):

		try:
			curr_time = datetime.datetime.now()

			# this is appended later when writing to the .csv file, if the validation set is not running, move to the next line
			if validation == True:
				valid_string = ''
			else: valid_string = '\n'

			# load selected data set
			if data_set == 'CIFAR10':
				training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_CIFAR_10(data_location, validation, test_split)
				self.regression = False
				self.show_classification_key.emit(self.regression)
			if data_set == 'MNIST':
				training_set, training_labels, validation_set, validation_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_MNIST(data_location, validation, test_split)
				self.regression = False
			if data_set == 'PrimaHeadPosePitch':
				training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_prima_head_pose_pitch(data_location, prima_test_person_out)
				self.regression = True
			if data_set == 'PrimaHeadPoseYaw':
				training_set, training_labels, testing_set, testing_labels, image_size, num_channels, num_classes = data.load_prima_head_pose_yaw(data_location, prima_test_person_out)
				self.regression = True


			if self.regression == False:
				self.change_accuracy_graph_axis.emit(self.regression, 'Accuracy (%)')
				self.change_loss_graph_axis.emit(self.regression, 'Loss')

			else: 
				self.change_accuracy_graph_axis.emit(self.regression, 'Mean Absolute Error')
				self.change_loss_graph_axis.emit(self.regression, 'Root Mean Squared Error')
				self.show_classification_key.emit(self.regression)

			self.change_graph_titles.emit(self.regression)


			# create .csv file dependent upon selected dataset
			if write_to_csv == True:
				active_csv_path = self.create_csv_file(data_set, csv_save_path, curr_time)
			else: active_csv_path = ''

			# take some random images from the test set to use for sample visualizations
			num_rows, _ = testing_labels.shape
			rand_num = np.random.randint(0, num_rows, (9))
			sample_images = np.take(testing_set, rand_num, axis=0, out=None)
			sample_labels = np.take(testing_labels, rand_num, axis=0, out=None)
			sample_labels = (sample_labels * 180) - 90
			if self.regression == False:
				sample_labels = np.argmax(sample_labels, axis=1)
			# generate sample image visualizations
			sample_file_name = self.plot_sample_images(vis_save_path, data_set, sample_images, sample_labels)
			# send file path to GUI to load into application
			self.embed_sample_images.emit(sample_file_name, False)

			
		# if dataset is not found in location found in Settings
		except FileNotFoundError:
			self.log_message.emit('File Not Found In Location, Please Select Valid Location in Settings')
			self.work_complete.emit()
			return 0
		# exception if MNIST is not found and it tries to download without internet connection
		except urllib.error.URLError:
			self.log_message.emit('MNIST Not Found In Location, No Internet Connection detected to download data set')
			self.work_complete.emit()
			return 0


		# perform simple normalization data, if user requires
		if normalize == True:
			training_set -= 127 
			testing_set -= 127
			if validation == True:
				validation_set -= 127

		try:
			# get array of layers from XML file
			layers = pp.get_layers(model_path)
		except FileNotFoundError:
			self.log_message.emit('Error Reading XML File. Could Not Locate XML File. Please Select a Valid File.')
			self.work_complete.emit()
			return 0
		except ET.ParseError:
			self.log_message.emit('Error Reading XML File. XML File is badly formed. Please Select a Valid File.')
			self.work_complete.emit()
			return 0
		except PermissionError:
			self.log_message.emit('Permission Denied Accessing Model .XML File. Please Select a Valid File.')
			self.work_complete.emit()
			return 0

		# send network model to GUI to print to log
		self.network_model.emit(layers)
		self.log_message.emit('Initialising Tensorflow Variables... \n')
		
		# initialise Tensorflwo computation graph
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

				# record initial input dimension for first layer
				input_dimension = image_size

				try:
					# create as many layers as are stored in XML file
					for e in range(len(layers)):
						# return newly created layer and its input dimension and feed to next layer
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

			self.log_message.emit('Batch Size: {0}'.format(str(batch_size)))
			if validation == True:
				self.log_message.emit('Test Set: {0}% : Validation Set {1}%'.format(str(test_split), str((100 - test_split))))

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
				self.log_message.emit('Optimizer: Momentum, Momentum Value: {0}'.format(str(momentum)))

			self.log_message.emit('Learning Rate: {0}\n'.format(str(learning_rate)))

			if self.regression == False:
				# convert one hot encoded array to single prediction value
				network_pred_class = tf.argmax(model_output, dimension=1)
				correct_prediction = tf.equal(network_pred_class, class_labels)
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
				# this value is used later in construction of confusion matrix vis
				confusion_classes = 10
			else: 
				network_pred_angle = model_output
				confusion_classes = 12


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
				for epoch in range(0, num_epochs + 1):
					# check if user has signalled training to be cancelled
					if self.end_training == False:
						# grab random batch from training data
						offset = (epoch * batch_size) % (training_labels.shape[0] - batch_size)
						batch_data = training_set[offset:(offset + batch_size), :, :, :]
						batch_labels = training_labels[offset:(offset + batch_size)]

						# configure placeholder variables
						feed_dict = {x: batch_data, y: batch_labels}

						# run computation depending if selected data set is classification or regression
						if self.regression == False:
							_, batch_loss, batch_accuracy, batch_pred_class, batch_classes = session.run([optimizer, loss, accuracy, network_pred_class, class_labels], feed_dict=feed_dict)
						else: 
							_, batch_loss, batch_pred_angles, batch_angles = session.run([optimizer, loss, model_output, angle_labels], feed_dict=feed_dict)

						# print information to output log
						if (epoch % self.update_interval == 0):
							self.log_message.emit('')
							self.log_message.emit('Global Step: {}'.format(str(global_step.eval())))
							self.log_message.emit('Loss at epoch: {} of {} is {}'.format(epoch, str(num_epochs), batch_loss))
							
							if self.regression == False:
								# calculate accuracy for classification task and send to log
								self.log_message.emit('Batch Accuracy = {}%'.format(str(batch_accuracy * 100)))
								# update batch accuracy graph for classification task
								self.batch_acc.emit((batch_accuracy * 100), epoch)
								# write batch information to .csv for classification task
								if write_to_csv == True:
									try:
										file = open(active_csv_path, 'a')
										file.write('Batch ' + str(epoch) + ',' + str(batch_loss) + ',' + str(batch_accuracy * 100) + '%' + valid_string)
										file.close()
									except PermissionError:
										self.log_message.emit('Permission Denied opening log file, please close the file or change the permissions')

							else: 
								# evaluate accuracy of regression task and send to output log
								RMSE, RMSE_stdev, MAE, MAE_stdev = self.evaluate_accuracy(batch_pred_angles, batch_labels, False, '')
								batch_classes, batch_pred_class = self.convert_regression_to_classification(batch_pred_angles, batch_angles)
								correct_prediction = np.equal(batch_classes, batch_pred_class)
								batch_accuracy = np.mean(correct_prediction.astype(float))
								self.log_message.emit('Batch Root Mean Squared Error = {}'.format(str(RMSE)))
								self.log_message.emit('Batch Root Mean Squared Error Standard Deviation = {}'.format(str(RMSE_stdev)))
								self.log_message.emit('Batch Mean Absolute Error = {}'.format(str(MAE)))
								self.log_message.emit('Batch Mean Absolute Error Standard Deviation = {}'.format(str(MAE_stdev)))
								self.log_message.emit('Batch Accuracy (Split into 12 discrete classes of 15 degrees each) = {}%'.format(str(batch_accuracy * 100)))

								# write batch information to .csv for regression task
								if write_to_csv == True:
									try:
										file = open(active_csv_path, 'a')
										file.write('Batch ' + str(epoch) + ',' + str(batch_loss) + ',' + str(RMSE) + ',' + str(RMSE_stdev) + ',' + str(MAE) + ',' + str(MAE_stdev) + '\n')
										file.close()
									except PermissionError:
										self.log_message.emit('Permission Denied opening log file, please close the file or change the permissions')

								# send batch accuracy to update accuracy graph (MAE for regression)
								self.batch_acc.emit((batch_accuracy * 100), epoch)
								self.train_valid_acc.emit(RMSE, 0, epoch)
								self.train_valid_loss.emit(MAE, 0, epoch)
							
							# emit batch loss for GUI visualization
							self.batch_loss.emit(batch_loss, epoch)
							

							# if user has chosen to include run time visualizations
							if run_time == True:

								if train_confusion_active == True:

									# create confusion matrix for current batch
									batch_confusion = tf.contrib.metrics.confusion_matrix(batch_classes, batch_pred_class, confusion_classes).eval()
									file_name = self.create_confusion_matrix(batch_confusion, vis_save_path, True, epoch)
									self.confusion_mat.emit(file_name, epoch, True)

								if validation == True:

									# split training data into batches and feed into network to reduce amount of GPU VRAM required
									num_training_data = len(training_labels)
									# split training data into batches and feed through network
									data_per_batch = ceil(num_training_data / 100)
									num_batches = num_training_data / data_per_batch
									# initialise total training accuracy and loss variables
									train_accuracy_total = 0
									train_loss_total = 0

									# cycle all batches and record/summate accuracy
									for i in range(int(num_batches)):
									   batch_data = training_set[i*data_per_batch:(i+1)*data_per_batch]
									   batch_labels = training_labels[i*data_per_batch:(i+1)*data_per_batch]
									   train_accuracy, train_loss = session.run([accuracy, loss], feed_dict={x: batch_data, y: batch_labels})
									   train_accuracy_total += train_accuracy
									   train_loss_total += train_loss
									# calculate mean across entire training set
									train_accuracy_total /= num_batches
									train_loss_total /= num_batches


									# split validation data into batches and feed into network to reduce amount of GPU VRAM required
									num_valid_data = len(validation_labels)
									# split training data into batches and feed through network
									data_per_batch = ceil(num_valid_data / 10)
									num_batches = num_valid_data / data_per_batch
									# initialise total training accuracy and loss variables
									valid_accuracy_total = 0
									valid_loss_total = 0

									# cycle all batches and record/summate accuracy
									for i in range(int(num_batches)):
									   batch_data = validation_set[i*data_per_batch:(i+1)*data_per_batch]
									   batch_labels = validation_labels[i*data_per_batch:(i+1)*data_per_batch]
									   valid_accuracy, valid_loss = session.run([accuracy, loss], feed_dict={x: batch_data, y: batch_labels})
									   valid_accuracy_total += valid_accuracy
									   valid_loss_total += valid_loss
									# calculate mean across entire training set
									valid_accuracy_total /= num_batches
									valid_loss_total /= num_batches

									# write validation set accuracy into .csv file if required
									if write_to_csv == True:
										try:
											file = open(active_csv_path, 'a')
											file.write(',,' + str(valid_loss_total) + ',' + str(valid_accuracy_total * 100) + '%\n')
											file.close()
										except PermissionError:
											self.log_message.emit('Permission Denied opening log file, please close the file or change the permissions')

									# send values to graphs to update	
									self.train_valid_acc.emit((train_accuracy_total * 100), (valid_accuracy_total * 100), epoch)
									self.train_valid_loss.emit(train_loss_total, valid_loss_total, epoch)


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
							except tf.errors.UnknownError:
								self.log_message.emit('Directory to save checkpoints to does not exist. Please Change. Checkpoint NOT Saved.')


				self.log_message.emit('\nTraining Complete')

				# calculate time taken to train and print to log
				train_end = datetime.datetime.now()
				training_time = train_end - train_start
				self.log_message.emit(('Training Took ' + str(training_time)) + '\n')

				try:
					# save network state when complete
					saver.save(sess=session, save_path=save_path, global_step=global_step)
					self.log_message.emit('Saved Checkpoint')
				except ValueError:
					self.log_message.emit('Directory to save checkpoints to does not exist. Please Change. Checkpoint NOT Saved.')
				except tf.errors.UnknownError:
					self.log_message.emit('Directory to save checkpoints to does not exist. Please Change. Checkpoint NOT Saved.')

				# set progress bar to 100% (if training was not interrupted)
				if (self.end_training == False):
					self.epoch_progress.emit(100)

				self.log_message.emit('\nEvaluating Test Set...')


				# run test set through network
				if self.regression == False:

					# split testing data into batches and feed into network to reduce amount of GPU VRAM required
					# this procedure is only really relevant to MNIST and CIFAR due to the number of data present

					num_test_data = len(testing_labels)
					# split training data into batches and feed through network
					data_per_batch = ceil(num_test_data / 10)
					num_batches = num_test_data / data_per_batch

					predicted_classes = []

					# initialise total training accuracy and loss variables
					testing_accuracy_total = 0

					# cycle all batches and record/summate accuracy
					for i in range(int(num_batches)):
					   batch_data = testing_set[i*data_per_batch:(i+1)*data_per_batch]
					   batch_labels = testing_labels[i*data_per_batch:(i+1)*data_per_batch]
					   test_accuracy, testing_pred_class = session.run([accuracy, network_pred_class], feed_dict={x: batch_data, y: batch_labels})
					   testing_accuracy_total += test_accuracy
					   predicted_classes = np.append(predicted_classes, testing_pred_class)

					# calculate mean across entire training set
					testing_accuracy_total /= num_batches
					actual_classes = np.argmax(testing_labels, axis=1)


					if write_to_csv == True:
						# write test set performance to .csv for clssification task
						num_rows, _ = testing_labels.shape
						file = open(active_csv_path, 'a')
						file.write('\n\nTesting Set Performance\nTest Accuracy = ' + str(testing_accuracy_total * 100) + ' %\n')
						file.write('Predicted Class, Actual Class\n')
						for i in range(0, num_rows):
							file.write(str(predicted_classes[i]) + ',' + str(actual_classes[i]) + '\n')
						file.close()

					# print test set accuracy to log
					self.log_message.emit('\nTest Set Accuracy = {}%'.format(str(testing_accuracy_total * 100)))

					# generate visualization of sample images and network prediction for classification task
					sample_preds = np.take(predicted_classes, rand_num, axis=0, out=None)
					sample_file_name = self.plot_sample_images(vis_save_path, data_set, sample_images, sample_labels, sample_preds)
					self.embed_sample_images.emit(sample_file_name, True)
					
				else: 
					# otherwise calculate evaluate accuracy for regression task, use entire test set
					pred_angles = session.run(model_output, feed_dict={x: testing_set, y: testing_labels})
					RMSE, RMSE_stdev, MAE, MAE_stdev = self.evaluate_accuracy(pred_angles, testing_labels, write_to_csv, active_csv_path)
					# if prima task is selected, convert angles into 12 discrete classes
					predicted_classes, actual_classes = self.convert_regression_to_classification(pred_angles, testing_labels)
					correct_prediction = np.equal(predicted_classes, actual_classes)
					test_accuracy = np.mean(correct_prediction.astype(float))

					# print test set performance
					self.log_message.emit('Test Set Root Mean Squared Error = {}'.format(str(RMSE)))
					self.log_message.emit('Test Set Root Mean Squared Error Standard Deviation = {}'.format(str(RMSE_stdev)))
					self.log_message.emit('Test Set Mean Absolute Error = {}'.format(str(MAE)))
					self.log_message.emit('Test Set Mean Absolute Error Standard Deviation = {}'.format(str(MAE_stdev)))
					self.log_message.emit('Test Set Accuracy (Split into 12 discrete classes of 15 degrees each) = {}%'.format(str(test_accuracy * 100)))

					# generate visualization of sample images and network prediction for regression task
					sample_preds = np.take(pred_angles, rand_num, axis=0, out=None)
					sample_preds = (sample_preds * 180) - 90
					sample_file_name = self.plot_sample_images(vis_save_path, data_set, sample_images, sample_labels, sample_preds)
					self.embed_sample_images.emit(sample_file_name, True)

				self.log_message.emit('Test Set Evaluated \n')
				self.log_message.emit('Loading Visualizations...')

				if run_time == True:

					if test_confusion_active == True:
						# create confusion matrix from predicted and actual classes
						test_set_confusion = tf.contrib.metrics.confusion_matrix(actual_classes, predicted_classes, confusion_classes).eval()
						file_name = self.create_confusion_matrix(test_set_confusion, vis_save_path, False, 0)
						self.confusion_mat.emit(file_name, 0, False)

					# list to contain names of convolution layers
					conv_layer_names = []

					# cycle all convolution layers and store names
					for layer in layers:
						if layer.layer_type == 'Convolution':
							conv_layer_names.append(layer.layer_name)

					# generate visualization of convolution layer weights (if user requested)
					if conv_weights_active == True:
						layer_weights_file_names = []
						# cycle all Tensorflow convolution layers and create plot, return file path
						for e in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
							if 'Cweights' in e.name:
								file_name = self.plot_conv_weights(weights=e, vis_save_path=vis_save_path, name=e.name, session=session)
								layer_weights_file_names.append(file_name)

						# send all file names to function that embeds convolution weights visualizations in application
						self.network_weights.emit(layer_weights_file_names, conv_layer_names)

					# generate visualization of convolution layer outputs (if user requested)
					if conv_outputs_active == True:
						layer_outputs_file_names = []
						
						# grab random image from test set
						random = np.random.randint(0, testing_set.shape[0])
						image = testing_set[random]
						layer_count = 0
						# create convolution output plots for each convolution layer, return directory of saved plot
						for layer in network_layers:
							if layers[layer_count].layer_type == 'Convolution':
								file_name = self.plot_conv_layer(layer=layer, vis_save_path=vis_save_path, name=layers[layer_count].layer_name, image=image, session=session, x=x)
								layer_outputs_file_names.append(file_name)
							layer_count += 1

						# send all file names to function that embeds convolution outputs visualizations in application
						self.network_outputs.emit(layer_outputs_file_names, conv_layer_names)	

				self.log_message.emit('Visualizations Loaded\n')


			# signal that thread has completed
			self.work_complete.emit()


	# called when user pressed the cancel button to stop training
	def cancel_training(self):
		self.end_training = True

	# creates .csv file with name dependent upon data set selected
	# 	@param data_set = string of the data set selected by user
	# 	@param csv_save_path = string of directory where to save .csv file on disk
	# 	@param curr_time = datetime object when program started
	def create_csv_file(self, data_set, csv_save_path, curr_time):
			active_csv_path = csv_save_path + data_set + '_outputlog_' + str(curr_time.hour) + str(curr_time.minute) + str(curr_time.second) + '.csv'
			file = open(active_csv_path, 'a')
			file.write(',Batch Loss,Batch Accuracy,,Validation Loss,Validation Accuracy\n')
			file.close()
			return active_csv_path

	# generates sample images 
	#	@param vis_save_path = string of directory where to save .csv file on disk
	# 	@param data_set = string of the data set selected by user
	# 	@param images = numpy array containing sample images
	# 	@param actual_labels = numpy array containing sample images actual labels
	#	@param predicted_labels = numpy array containing sample images predicted images
	def plot_sample_images(self, vis_save_path, data_set, images, actual_labels, predicted_labels=None):

		# change the format of the plot, some plots work better with and without this 
		rcParams.update({'figure.autolayout': False})

		try:
		   # create figure and number of subplots within
			fig, axes = plt.subplots(3, 3)
			# configure the size and spacing of subplots
			fig.subplots_adjust(left=0.15, bottom=0.1, right=0.85, top=0.87)
			# if cifar is selected, generate the class names to display
			if data_set == 'CIFAR10':
				class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


			# if this is the first time, perform normalization of images so they appear normal
			if predicted_labels is None:
				images /= 255
			else: 
				if not 'Prima' in data_set:
					predicted_labels = predicted_labels.astype(int)

			# for each subplot
			for i, ax in enumerate(axes.flat):

				# plot image
				if data_set == 'MNIST':
					# reshape if MNIST, it doesnt like the 1 channel format
					ax.imshow(images[i].reshape((28, 28)), cmap='binary')
				else:
					ax.imshow(images[i, :, :, :], interpolation='nearest')
					
				# if no predicted labels were supplied
				if predicted_labels is None:
					if data_set == 'CIFAR10':
						x_axis_title = "Actual: {0}".format(class_names[actual_labels[i]])
					else: x_axis_title = "Actual: {0}".format(np.array_str(actual_labels[i], precision=3))
					# to be used when saving the figure to disk
					save_name = 'samples'
				else:
					if data_set == 'CIFAR10':
						x_axis_title = "Predicted: {0}".format(class_names[predicted_labels[i]])
					else: x_axis_title = "Predicted: {0}".format(np.array_str(predicted_labels[i], precision=3))
					# to be used when saving the figure to disk
					save_name = 'predictions'

				# set a axis title to actual of preiction class
				ax.set_xlabel(x_axis_title)
				
				# Remove ticks from the plot
				ax.set_xticks([])
				ax.set_yticks([])
			
			# save to disk and return file name
			file_name = vis_save_path + save_name + ".png"
			plt.savefig(file_name, format='png')
			plt.close()
			return file_name

		except FileNotFoundError:
			self.log_message.emit('Directory chosen to save visualizations was not found. Please amend in Settings.')


	# function that creates the image of convolution layer outputs
	#	@param layer = Tensorflow layer to pass image through
	# 	@param vis_save_path = string containing where to save visualization
	# 	@param name = string name of convolution layer for saving
	# 	@param image = numpy array of image to be passed through layer
	#	@param session = reference of the Tensorflow session to allow run
	#	@param session = reference to placeholder variable from run to insert image into
	def plot_conv_layer(self, layer, vis_save_path, name, image, session, x):

		# change the format of the plot, some plots work better with and without this 
		rcParams.update({'figure.autolayout': False})
		try:
			# feed image through network
			feed_dict = {x: [image]}

			# Calculate and retrieve the output values of the layer
			values = session.run(layer, feed_dict=feed_dict)

			values_min = np.min(values)
			values_max = np.max(values)

			# Number of filters used in the convolution. layer.
			num_filters = values.shape[3]

			# Number of grids to plot.
			num_grids = ceil(sqrt(num_filters))
			
			# Create figure with a grid of sub-plots.
			fig, axes = plt.subplots(num_grids, num_grids)

			# format size of plot to fit nicely in GUI
			fig.subplots_adjust(left=0.05, bottom=0.01, right=0.99, top=0.99)

			# Plot the output images of all the filters.
			for i, ax in enumerate(axes.flat):
				# Only plot the images for valid filters.
				if i < num_filters:
					# Get the output image of using the i'th filter.
					img = values[0, :, :, i]

					# Plot image.
					ax.imshow(img, vmin=values_min, vmax=values_max, interpolation='nearest', cmap='binary')
				
				# remove ticks from the plot
				ax.set_xticks([])
				ax.set_yticks([])

			# save figure to disk and return location so it can be loaded into application later
			file_name = vis_save_path + name + "_output.png"
			plt.savefig(file_name, format='png')

			plt.close()
			return file_name
		except FileNotFoundError:
			self.log_message.emit('Directory chosen to save visualizations was not found. Please amend in Settings.')

	# function that plots visualization of convolution layer weights
	#	@param weights = Tf.Trainable variable containing weights of layer to visualize
	# 	@param vis_save_path = string containing where to save visualization
	# 	@param name = string name of convolution layer for saving
	#	@param session = reference of the Tensorflow session to allow run
	def plot_conv_weights(self, weights, vis_save_path, name, session):

		# change the format of the plot, some plots work better with and without this 
		rcParams.update({'figure.autolayout': False})
		try:
			# run a session using the designated convolution layer weights
			w = session.run(weights)

			w_min = np.min(w)
			w_max = np.max(w)
			abs_max = max(abs(w_min), abs(w_max))

			# number of filters used in the convolution layer
			num_filters = w.shape[3]

			# number of grids to plot
			num_grids = ceil(sqrt(num_filters))
			
			# create figure and subplots within
			fig, axes = plt.subplots(num_grids, num_grids)

			# format size of plot to fit nicely in GUI
			fig.subplots_adjust(left=0.05, bottom=0.01, right=0.99, top=0.99)

			# Plot all the filter-weights.
			for i, ax in enumerate(axes.flat):
				# Only plot the valid filter-weights.
				if i < num_filters:
					# get the weights for the i'th filter of the input channel
					img = w[:, :, 0, i]

					# plot image
					ax.imshow(img, vmin=-abs_max, vmax=abs_max, interpolation='nearest', cmap='seismic')
				
				# remove ticks from the plot.
				ax.set_xticks([])
				ax.set_yticks([])

			# save figure to disk and return location so it can be loaded into application later
			name = name[:-2]
			file_name = vis_save_path + name + ".png"
			plt.savefig(file_name, format='png')

			plt.close()
			return file_name

		except FileNotFoundError:
			self.log_message.emit('Directory chosen to save visualizations was not found. Please amend in Settings.')

	# function that creates confusion matrix visualization
	#	@param confusion = confusion matrix created by tf.contrib.metrics.confusion_matrix()
	# 	@param vis_save_path = string containing where to save visualization
	# 	@param training = bool that says whether this is for a training batch or the testing set
	#	@param batch = int value of which batch/epoch this is
	def create_confusion_matrix(self, confusion, vis_save_path, training, batch):

		# change the format of the plot, some plots work better with and without this 
		rcParams.update({'figure.autolayout': True})

		try:
			# create figure and axes
			self.fig = plt.figure()
			self.axes = self.fig.add_subplot(111)
			self.axes.set_aspect(1)
				
			# create plot
			res = self.axes.imshow(np.array(confusion), cmap=plt.cm.jet, interpolation='nearest')
			
			width, height = confusion.shape

			# cycle all elements in array and plot on corresponding axes
			for x in range(width):
				for y in range(height):
					self.axes.annotate(str(confusion[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

			# add colour bar to visualizatoin
			cb = self.fig.colorbar(res)

			# labels to print on each axes
			alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']

			# set axes to 0-9
			plt.xticks(range(width), alphabet[:width])
			plt.yticks(range(height), alphabet[:height])
			# add axis titles
			plt.xlabel('Predicted')
			plt.ylabel('Actual')

			# save figure to disk so it can be loaded later
			if training == True:
				file_name = vis_save_path + 'training_confusion_matrix_batch_' + str(batch) + '.png'
				plt.savefig(file_name, format='png')
			else:
				file_name = vis_save_path + 'testing_confusion_matrix.png'
				plt.savefig(file_name, format='png')

			plt.close()
			return file_name

		# sometimes get a division by zero error, this just skips creating that matrix.. very rare occurance
		except ZeroDivisionError:
			do_nothing = 0
		except FileNotFoundError:
			self.log_message.emit('Directory chosen to save visualizations was not found. Please amend in Settings.')

	# this function converts the predicted and actual labels (in degrees) into 12 discrete classes, with 15 degress separating each
	# it returns two new arrays that contain a class value for each predicted and actual label
	# 	@param predictions = numpy array containing predicted angles from the network
	# 	@param labels = numpy array containing actual angles
	def convert_regression_to_classification(self, predictions, labels):
		# Convert prediction to degrees
		prediction_degree = (predictions * 180) - 90
		actual_degree = (labels * 180) - 90

		num_of_labels = len(labels)

		prediction_class = np.zeros([num_of_labels])
		actual_class = np.zeros([num_of_labels])

		for i in range(0, num_of_labels):
			if prediction_degree[i] < -75:
				prediction_class[i] = 0
			elif prediction_degree[i] < -60:
				prediction_class[i] = 1
			elif prediction_degree[i] < -45:
				prediction_class[i] = 2
			elif prediction_degree[i] < -30:
				prediction_class[i] = 3
			elif prediction_degree[i] < -15:
				prediction_class[i] = 4
			elif prediction_degree[i] < 0:
				prediction_class[i] = 5
			elif prediction_degree[i] < 15:
				prediction_class[i] = 6
			elif prediction_degree[i] < 30:
				prediction_class[i] = 7
			elif prediction_degree[i] < 45:
				prediction_class[i] = 8
			elif prediction_degree[i] < 60:
				prediction_class[i] = 9
			elif prediction_degree[i] < 75:
				prediction_class[i] = 10
			elif prediction_degree[i] < 90:
				prediction_class[i] = 11

			if actual_degree[i] < -75:
				actual_class[i] = 0
			elif actual_degree[i] < -60:
				actual_class[i] = 1
			elif actual_degree[i] < -45:
				actual_class[i] = 2
			elif actual_degree[i] < -30:
				actual_class[i] = 3
			elif actual_degree[i] < -15:
				actual_class[i] = 4
			elif actual_degree[i] < 0:
				actual_class[i] = 5
			elif actual_degree[i] < 15:
				actual_class[i] = 6
			elif actual_degree[i] < 30:
				actual_class[i] = 7
			elif actual_degree[i] < 45:
				actual_class[i] = 8
			elif actual_degree[i] < 60:
				actual_class[i] = 9
			elif actual_degree[i] < 75:
				actual_class[i] = 10
			elif actual_degree[i] < 90:
				actual_class[i] = 11

		return prediction_class, actual_class



# this is a simple custom canvas that can be embedded directly into a PyQt Widget. Inherits from the FigureCanvas class.
class MyMplCanvas(FigureCanvas):

	def __init__(self, parent=None, width=20, height=20, dpi=100, x_axis_title='x', y_axis_title='y'):
		self.x_axis_title = x_axis_title
		self.y_axis_title = y_axis_title

		# create figure and subplot within
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.axes = self.fig.add_subplot(111)

		# format size of plot to fit nicely in GUI
		self.fig.subplots_adjust(left=0.05, bottom=0.01, right=0.99, top=0.99)

		# format size of plot to fit nicely in GUI
		self.fig.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.87)

		# add titles to both axis
		self.axes.set_xlabel(self.x_axis_title)
		self.axes.set_ylabel(self.y_axis_title)

		# add grid to plot
		self.axes.grid(True)

		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)

		FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)


# this graph gets updated within the application, it draws a single line
class DynamicGraph(MyMplCanvas):
	def __init__(self, *args, **kwargs):
		MyMplCanvas.__init__(self, *args, **kwargs)
		self.errors = []
		self.epochs = []

	@pyqtSlot(float, int)
	def update_figure(self, error: float, epoch: int):
		
		# if this is the first run, clear the axis and reset the lists
		if epoch == 0:
			self.axes.clear()
			self.axes.set_xlabel(self.x_axis_title)
			self.axes.set_ylabel(self.y_axis_title)
			self.axes.grid(True)
			self.errors = []
			self.epochs = []

		# add current values to array
		self.errors.append(error)
		self.epochs.append(epoch)

		# plot new, extended graph
		self.axes.plot(self.epochs, self.errors, 'r')
		
		self.draw()


# this graph is is the same as above except with two lines, for validation/training accuracy
class DynamicDoubleGraph(MyMplCanvas):
	def __init__(self, *args, **kwargs):
		MyMplCanvas.__init__(self, *args, **kwargs)

		self.regression_mode = False
		self.train_errors = []
		self.valid_errors = []
		self.epochs = []
		self.legend = self.axes.legend(loc='upper center', shadow=True)

	@pyqtSlot(float, float, int)
	def update_figure(self, train_error: float, valid_error: float, epoch: int):

		if epoch == 0:
			self.axes.clear()
			self.axes.set_xlabel(self.x_axis_title)
			self.axes.set_ylabel(self.y_axis_title)
			self.axes.grid(True)
			self.train_errors = []
			self.valid_errors = []
			self.epochs = []

		# add current values to array
		self.train_errors.append(train_error)
		self.valid_errors.append(valid_error)
		self.epochs.append(epoch)

		# re plot the graph, incorporating new extended arrays
		self.axes.plot(self.epochs, self.train_errors, 'r', label="train")
		if self.regression_mode == False:
			self.axes.plot(self.epochs, self.valid_errors, 'blue', label="valid")
		self.draw()

	@pyqtSlot(bool, str)
	def update_graph_axis(self, regression: bool, axis_title: str):
		if regression == True:
			self.regression_mode = True
		else:
			self.regression_mode = False

		self.y_axis_title = axis_title
		self.axes.set_ylabel(self.y_axis_title)

# The GUI framework created in Qt Designer
class CNNApp(QMainWindow, design.Ui_MainWindow):

	# signal to background thread that the user has requested training to cancel
	end_train = pyqtSignal()

	def __init__(self):  
		super().__init__()
		self.setupUi(self)

		# initialise threads array, if multiple required
		self.__threads = None

		# stores the layers added when user creates a new model
		self.new_model = []
		# stores the layer names added when user creates a new model, stops multiple of the same name being added
		self.layer_names = []

		# connect navigational icon  methods (that change which tab is open)
		self.actionTrain.triggered.connect(partial(self.open_tab, index=0))
		self.actionDesign.triggered.connect(partial(self.open_tab, index=1))
		self.radCreateModel.toggled.connect(self.create_model_rad_clicked)  # similar to above, takes to design page but also deselects radio button
		self.actionVisualizations.triggered.connect(partial(self.open_tab, index=2))
		self.actionSettings.triggered.connect(partial(self.open_tab, index=3))
		self.actionExit.triggered.connect(self.close_event)

		# fills checkpoint save path to same location as load
		self.cbxSavePath.stateChanged.connect(self.checkpoint_checkbox_state_changed)
		
		# connect test/validation split spinner function that updates validation value dependint on change to test ratio
		self.spnTestSplit.valueChanged.connect(self.testing_validation_split_spinner_changed)

		# events for checking/unchecking dropout when creating model
		self.chkConvDropout.stateChanged.connect(partial(self.dropout_checkbox_state_changed, dropout_checkbox=self.chkConvDropout, keep_rate_spinner=self.spnConvKeepRate))
		self.chkPoolDropout.stateChanged.connect(partial(self.dropout_checkbox_state_changed, dropout_checkbox=self.chkPoolDropout, keep_rate_spinner=self.spnPoolKeepRate))
		self.chkFCDropout.stateChanged.connect(partial(self.dropout_checkbox_state_changed, dropout_checkbox=self.chkFCDropout, keep_rate_spinner=self.spnFCKeepRate))

		# hides or shows momentum spinner when choosing momentum optimizer
		self.cbxOptimizer.currentIndexChanged.connect(self.optimizer_combo_box_changed)
		# makes beta value enabled or disabled when selecting whether to use L2 reg
		self.chkL2Reg.stateChanged.connect(self.l2_reg_checkbox_state_changed)

		# changes the text field and hides/shows spinner when choosing method of initialization for weights
		self.cbxConvWeightInit.currentIndexChanged.connect(partial(self.weight_init_combo_changed, weight_init_combobox=self.cbxConvWeightInit, 
			weight_val_spinner= self.spnConvStdDev, weight_val_label= self.lblConvStdDev, help_icon=self.hlpConvStdDev))
		self.cbxFCWeightInit.currentIndexChanged.connect(partial(self.weight_init_combo_changed, weight_init_combobox=self.cbxFCWeightInit, 
			weight_val_spinner= self.spnFCStdDev, weight_val_label= self.lblFCStdDev, help_icon=self.hlpFCStdDev))
		self.cbxOutputWeightInit.currentIndexChanged.connect(partial(self.weight_init_combo_changed, weight_init_combobox=self.cbxOutputWeightInit, 
			weight_val_spinner= self.spnOutputStdDev, weight_val_label= self.lblOutputStdDev, help_icon=self.hlpOutputStdDev))

		# changes the text field and hides/shows spinner when choosing method of initialization for biases
		self.cbxConvBiasInit.currentIndexChanged.connect(partial(self.bias_init_combo_changed, bias_init_combobox=self.cbxConvBiasInit, 
			bias_val_spinner= self.spnConvBiasVal, bias_val_label= self.lblConvBiasVal, help_icon=self.hlpConvBiasVal))
		self.cbxFCBiasInit.currentIndexChanged.connect(partial(self.bias_init_combo_changed, bias_init_combobox=self.cbxFCBiasInit, 
			bias_val_spinner= self.spnFCBiasVal, bias_val_label= self.lblFCBiasVal, help_icon=self.hlpFCBiasVal))
		self.cbxOutputBiasInit.currentIndexChanged.connect(partial(self.bias_init_combo_changed, bias_init_combobox=self.cbxOutputBiasInit, 
			bias_val_spinner= self.spnOutputBiasVal, bias_val_label= self.lblOutputBiasVal, help_icon=self.hlpOutputBiasVal))

		# buttons to change file paths for various fields
		self.btnChangeSavePath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtSavePath))
		self.btnChangeLoadCheckpoints.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtLoadCheckpoints, disable=True))
		self.btnChangeModelPath.clicked.connect(self.change_file_path)
		self.btnChangeModelSavePath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtModelSavePath))
		self.btnChangeMNISTPath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtMNISTPath))
		self.btnChangeCIFARPath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtCIFARPath))
		self.btnChangePrimaPitchPath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtPrimaPitchPath))
		self.btnChangePrimaYawPath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtPrimaYawPath))
		self.btnChangeVisPath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtVisSavePath))
		self.btnChangeCSVPath.clicked.connect(partial(self.change_directory_path, path_text_field=self.txtCSVSavePath))
		
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
		self.batch_loss_graph = DynamicGraph(self.grphBatchLoss, width=5, height=4, dpi=100, x_axis_title='Epoch', y_axis_title='Loss')
		self.batch_acc_graph = DynamicGraph(self.grphBatchAcc, width=5, height=4, dpi=100, x_axis_title='Epoch', y_axis_title='Accuracy (%)')
		self.train_valid_accuracy = DynamicDoubleGraph(self.grphTrainValidAcc, width=5, height=4, dpi=100, x_axis_title='Epoch', y_axis_title='Accuracy (%)')
		self.train_valid_loss = DynamicDoubleGraph(self.grphTrainValidLoss, width=5, height=4, dpi=100, x_axis_title='Epoch', y_axis_title='Loss')

	# this function is triggered when the user commences training
	def train_button_clicked(self):
		# clear visualizations of previous runs
		self.reset_visualizations()

		try:
			# read input from data entry fields on GUI
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

			# set various variables from user input to be user in training phase
			if self.chkValidationSet.isChecked():
				validation = True
				test_split = int(self.spnTestSplit.text())
			else: 
				validation = False
				test_split = 0

			if self.cbxOptimizer.currentIndex() == 4:
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

			if self.chkWriteToCSV.isChecked():
				write_to_csv = True
				csv_save_path = self.txtCSVSavePath.text()
			else:
				write_to_csv = False
				csv_save_path = ''
	
		except ValueError:
			self.txtOutputLog.append('Number of Epochs, Batch Size and Learning Rate, Momentum and L2 Beta must be a Numerical Value!')
		else:
			# initialise threads array, suits multiple threads, but the application only uses one additional
			self.__threads = []

			# update GUI elements
			self.btnTrainNetwork.setDisabled(True)
			self.btnCancelTraining.setEnabled(True)
			self.prgTrainingProgress.setValue(0)

			# create worker object and thread
			worker = Worker(self.current_data_set(), data_location, prima_test_person_out, validation, test_split, num_epochs, batch_size, learning_rate, 
				momentum, optimizer, normalize, l2_reg, beta, save_path, save_interval, update_interval, model_path, vis_save_path, run_time, train_confusion_active, 
				test_confusion_active, conv_weights_active, conv_outputs_active, write_to_csv, csv_save_path)

			thread = QThread()

			# connect cancel button in main thread to background thread
			self.end_train.connect(worker.cancel_training)

			# store reference to objects so they are not garbage collected
			self.__threads.append((thread, worker))
			# move worker object to thread to commence work
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
			worker.embed_sample_images.connect(self.embed_sample_images)

			worker.change_graph_titles.connect(self.update_graph_titles)
			worker.change_loss_graph_axis.connect(self.train_valid_loss.update_graph_axis)
			worker.change_accuracy_graph_axis.connect(self.train_valid_accuracy.update_graph_axis)

			# connect the function that shows the cifar class key if selected
			worker.show_classification_key.connect(self.show_classification_key)

			# connect and start thread
			thread.started.connect(worker.work)
			thread.start()

	# function called when user tries to create a new convolution layer for new model
	def create_conv_layer_button_clicked(self):
		# check layer has a name
		if not self.txtConvName.text():
			self.txtOutputModelLog.append('Layer Must have a Name!')
			return 0
		# check new layer name is not already used
		if self.txtConvName.text() in self.layer_names:
			self.txtOutputModelLog.append('Cannot have duplicate layer names!')
			return 0

		try:
			# take input from data entry fields
			conv_layer_name = self.txtConvName.text()

			if self.cbxConvKernelSize.currentIndex() == 0:
				conv_kernel_size = 3
			elif self.cbxConvKernelSize.currentIndex() == 1:
				conv_kernel_size = 5
			elif self.cbxConvKernelSize.currentIndex() == 2:
				conv_kernel_size = 7

			conv_stride = self.return_conv_stride(self.cbxConvStride)
			num_output_filters = int(self.spnConvOutputFilters.text())
			act_function = self.return_act_function(self.cbxConvActFunction)
			weight_init, weight_std_dev = self.return_weight_init(self.cbxConvWeightInit, self.spnConvStdDev)
			bias_init, bias_val = self.return_bias_init(self.cbxConvBiasInit, self.spnConvBiasVal)
			padding = self.return_padding(self.cbxConvPadding)
			normalize = self.return_normalize(self.chkConvNorm)
			dropout, keep_rate = self.return_dropout(self.chkConvDropout, self.spnConvKeepRate)

			# add name to list of layer names to be checked
			self.layer_names.append(conv_layer_name)

			# create layer object that can be easily converted into .xml format
			layer = l.ConvLayer(conv_layer_name, conv_kernel_size, conv_stride, act_function, num_output_filters, weight_init, weight_std_dev, bias_init, bias_val, padding, normalize, dropout, keep_rate)
			# add to list of layers of new model
			self.new_model.append(layer)

			# create list item to be added to QListView object
			item = QListWidgetItem(("Convolution, Num Output Filters {}, Kernel Size: {}, Stride: {}, Activation Function: {}, Padding: {}, Normalize: {}, Dropout: {}, Keep Rate: {}").format(layer.num_output_filters, layer.kernel_size, layer.stride, layer.act_function, layer.padding, layer.normalize, layer.dropout, layer.keep_rate))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	# function called when user tries to add a max pooling layer to new model
	def create_pooling_layer_button_clicked(self):
		# check layer has a name
		if not self.txtPoolName.text():
			self.txtOutputModelLog.append('Layer Must have a Name!')
			return 0
		# check new layer name is not already used
		if self.txtPoolName.text() in self.layer_names:
			self.txtOutputModelLog.append('Cannot have duplicate layer names!')
			return 0

		try:    
			# take input from data entry fields
			pool_layer_name = self.txtPoolName.text()

			if self.cbxPoolKernelSize.currentIndex() == 0:
				pool_kernel_size = 2
			elif self.cbxPoolKernelSize.currentIndex() == 1:
				pool_kernel_size = 3
			elif self.cbxPoolKernelSize.currentIndex() == 2:
				pool_kernel_size = 4
			elif self.cbxPoolKernelSize.currentIndex() == 3:
				pool_kernel_size = 5

			pool_stride = self.return_pool_stride(self.cbxPoolStride)
			padding = self.return_padding(self.cbxPoolPadding)
			normalize = self.return_normalize(self.chkPoolNorm)
			dropout, keep_rate = self.return_dropout(self.chkPoolDropout, self.spnPoolKeepRate)

			# add name to list of layer names to be checked
			self.layer_names.append(pool_layer_name)

			# create layer object that can be easily converted into .xml format
			layer = l.MaxPoolingLayer(pool_layer_name, pool_kernel_size, pool_stride, padding, normalize, dropout, keep_rate)
			self.new_model.append(layer)

			# create list item to be added to QListView object
			item = QListWidgetItem(("Max Pool, Kernel Size: {}, Stride: {}, Padding: {}, Normalize: {}, Dropout: {}, Keep Rate: {}").format(layer.kernel_size, layer.stride, layer.padding, layer.normalize, layer.dropout, layer.keep_rate))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	# function called when user tries to add fully connected layer to new model
	def create_full_conn_layer_button_clicked(self):
		# check layer has a name
		if not self.txtFCName.text():
			self.txtOutputModelLog.append('Layer Must have a Name!')
			return 0
		# check new layer name is not already used
		if self.txtFCName.text() in self.layer_names:
			self.txtOutputModelLog.append('Cannot have duplicate layer names!')
			return 0

		try:
			# take input from data entry fields
			FC_layer_name = self.txtFCName.text()
			num_output_nodes = int(self.spnFCNumOutputNodes.text())
			act_function = self.return_act_function(self.cbxFCActFunction)
			weight_init, weight_std_dev = self.return_weight_init(self.cbxFCWeightInit, self.spnFCStdDev)
			bias_init, bias_val = self.return_bias_init(self.cbxFCBiasInit, self.spnFCBiasVal)
			dropout, keep_rate = self.return_dropout(self.chkFCDropout, self.spnFCKeepRate)

			# add name to list of layer names to be checked
			self.layer_names.append(FC_layer_name)

			# create layer object that can be easily converted into .xml format
			layer = l.FullyConnectedLayer(FC_layer_name, act_function, num_output_nodes, weight_init, weight_std_dev, bias_init, bias_val, dropout, keep_rate)
			self.new_model.append(layer)

			# create list item to be added to QListView object
			item = QListWidgetItem(("Fully Connected,  Num Output Nodes: {}, Activation Function: {}, Dropout: {}, Keep Rate: {}").format(layer.num_output_nodes, layer.act_function, layer.dropout, layer.keep_rate))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	# function called when user tries to add output layer to new model
	def create_output_layer_button_clicked(self):
		# check layer has a name
		if not self.txtOutputName.text():
			self.txtOutputModelLog.append('Layer Must have a Name!')
			return 0	
		# check new layer name is not already used
		if self.txtOutputName.text() in self.layer_names:
			self.txtOutputModelLog.append('Cannot have duplicate layer names!')
			return 0

		try:
			# take input from data entry fields
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

			# add name to list of layer names to be checked
			self.layer_names.append(output_layer_name)

			# create layer object that can be easily converted into .xml format
			layer = l.OutputLayer(output_layer_name, act_function, weight_init, weight_std_dev, bias_init, bias_val)
			self.new_model.append(layer)

			# create list item to be added to QListView object
			item = QListWidgetItem(("Output, Activation Function: {}").format(layer.act_function))
			self.lstModel.addItem(item)

		except ValueError:
			self.txtOutputModelLog.append('A field has been incorrectly entered, must be numbers!')

	# return convolution stride value depending on index of combobox
	# @param comboBox = reference to comboBox GUI element that contains stride value
	def return_conv_stride(self, comboBox):
		if comboBox.currentIndex() == 0:
			stride = 1
		elif comboBox.currentIndex() == 1:
			stride = 2
		elif comboBox.currentIndex() == 2:
			stride = 3
		return stride

	# return pooling stride value depending on index of combobox
	# @param comboBox = reference to comboBox GUI element that contains stride value
	def return_pool_stride(self, comboBox):
		if comboBox.currentIndex() == 0:
			stride = 2
		elif comboBox.currentIndex() == 1:
			stride = 3
		return stride

	# return activation function string depending on which index of combobox
	# @param comboBox = reference to comboBox GUI element that contains activation function to be used
	def return_act_function(self, comboBox):
		if comboBox.currentIndex() == 0:
			act_function = 'ReLu'
		elif comboBox.currentIndex() == 1:
			act_function = 'Sigmoid'
		elif comboBox.currentIndex() == 2:
			act_function = 'Tanh'
		return act_function

	# returns weight initialization configuration depending on combo box index and spinner value
	# @param comboBox = reference to comboBox GUI element that contains weight initialization method
	# @param spinner = reference to spinner GUI element to grab standard deviation from
	def return_weight_init(self, comboBox, spinner):
		if comboBox.currentIndex() == 0:
			weight_init = "Truncated Normal"
			weight_std_dev = float(spinner.text())
		elif comboBox.currentIndex() == 1:
			weight_init = "Random Normal"
			weight_std_dev = float(spinner.text())
		elif comboBox.currentIndex() == 2:
			weight_init = "Xavier"
			weight_std_dev = 0
		return weight_init, weight_std_dev

	# returns bias initialization configuration depending on combo box index and spinner value
	# @param comboBox = reference to comboBox GUI element that contains bias initialization
	# @param spinner = reference to spinner GUI element to grab bias STD/constant value from
	def return_bias_init(self, comboBox, spinner):
		if comboBox.currentIndex() == 0:
			bias_init = "Zeros"
			bias_val = 0
		elif comboBox.currentIndex() == 1:
			bias_init = "Random Normal"
			bias_val = float(spinner.text())
		elif comboBox.currentIndex() == 2:
			bias_init = "Truncated Normal"
			bias_val = float(spinner.text())
		elif comboBox.currentIndex() == 3:
			bias_init = "Constant"
			bias_val = float(spinner.text())
		return bias_init, bias_val

	# returns string stating which padding will be used during training
	# @param comboBox = reference to comboBox GUI element that says which form of padding to be used
	def return_padding(self, comboBox):
		if comboBox.currentIndex() == 0:
			padding = 'SAME'
		elif comboBox.currentIndex() == 1:
			padding = 'VALID'   
		return padding  

	# returns bool if normalization will be used in layer
	# @param checkBox = reference to checkBox GUI element that states whether to normalize layer
	def return_normalize(self, checkBox):
		if checkBox.isChecked():
			normalize = True
		else: normalize = False
		return normalize

	# returns bool if dropout will be used in layer and float of keep probability
	# @param checkBox = reference to checkBox GUI element that says whether dropout is used
	# @param spinner = reference to spinner GUI element to take keep rate value from
	def return_dropout(self, checkBox, spinner):
		if checkBox.isChecked():
			dropout = True
			keep_rate = float(spinner.text())
		else: 
			dropout = False
			keep_rate = 1.0
		return dropout, keep_rate

	# validates created model is functional for running, enables create model button if so
	def validate_model_button_clicked(self):
		# check model is not empty
		if not self.new_model:
			self.txtOutputModelLog.append('No Layers Added')
			return

		# check first layer is either a convolution or max pooling
		if self.new_model[0].layer_type != 'Convolution':
			self.txtOutputModelLog.append('First layer must be a Convolution layer')
			return

		# variables used to validate network
		fully_conn_count = 0
		conv_layer_count = 0
		no_more_convolution_or_pool = False
		output_has_occured = False

		for layer in self.new_model:
			if layer.layer_type == 'Fully Connected':
				fully_conn_count += 1
				no_more_convolution_or_pool = True
				# check output layer is only at the end of the model
				if output_has_occured == True:
					self.txtOutputModelLog.append('Output Layer can only be the final Layer in Model')
					return
			elif layer.layer_type == 'Convolution':
				conv_layer_count += 1
				if no_more_convolution_or_pool == True:
					# check user has not tried to go back to convolution steps after flattening input
					self.txtOutputModelLog.append('Cannot have Convolution Layer after a Fully Connected Layer')
					return
				if output_has_occured == True:
					self.txtOutputModelLog.append('Output Layer can only be the final Layer in Model')
					return
			elif layer.layer_type == 'Max Pool':
				if no_more_convolution_or_pool == True:
					# check user has not tried to go back to convolution steps after flattening input
					self.txtOutputModelLog.append('Cannot have a Max Pooling Layer after a Fully Connected Layer')
					return
				if output_has_occured == True:
					self.txtOutputModelLog.append('Output Layer can only be the final Layer in Model')
					return
			elif layer.layer_type == 'Output':
				if output_has_occured == True:
					self.txtOutputModelLog.append('Output Layer can only be the final Layer in Model')
					return
				output_has_occured = True

		# check atleast oen fully connected layer is present (required to flatten)
		if fully_conn_count == 0:
			self.txtOutputModelLog.append('Must have atleast one fully connected layer')
			return
		# check atleast one convolution layer is present (this is a CNN toolkit)
		elif conv_layer_count == 0:
			self.txtOutputModelLog.append('Must have atleast one Convolution layer')
			return

		# check the final layer of the model is an output layer
		if not self.new_model[-1].layer_type == 'Output':
			self.txtOutputModelLog.append('Final layer must be output layer')
			return

		# is passed all previous tests, allow user to save model
		self.txtOutputModelLog.append('Model Successfully Validated')
		self.btnCreateModel.setEnabled(True)

	# called when user presses create model button, should be disabled until user validates models
	def create_model_button_clicked(self):

		try:
			# check model name is present
			file_name = self.txtSaveModelAs.text()
			if not file_name:
				self.txtOutputModelLog.append("Please add name for new model")
				return False

			file_path = self.txtModelSavePath.text()

			# returns boolean if xml file was successfully created
			success = pp.create_XML_model(self.new_model, file_name, file_path)

			if success == True:
				self.txtOutputModelLog.append("Success Writing XML File")
				# allow user quick way to use new model
				response = QMessageBox.question(self, "Model Creation Successful", "Would you like to use the newly created model?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
				if response == QMessageBox.Yes:
					self.txtLoadModel.setText((file_path + file_name + '.xml'))
					self.tabPages.setCurrentIndex(0)
				# reset all model creation fields
				self.reset_model_creation()
			else : 
				self.txtOutputModelLog.append("Error Writing XML File")
				return False
		except:
			self.txtOutputModelLog.append("Error Writing XML File")
		
	# called to reset all fields in the model creation page
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
		self.layer_names = []
		self.grpClassKey.setFixedWidth(0)
		self.lblBatchConf.setText('Training Batch ')

	# called to reset visualizations when a new training run begins
	def reset_visualizations(self):
		self.lblSampleImages.setText('Some Sample Images from the Testing Set will Appear When Training Commences')
		self.lblSamplePredictions.setText('Some Sample Predictions will appear when the Testing Set has been Evaluated')
		self.lblTrainingConfusionMat.setText('Training Confusion Matrix for First Batch Will Appear When it has been Evaluated')
		self.lblTestingConfusionMat.setText('Test Set Confusion Matrix Will Appear When Test Set has been Evaluated')
		self.tabLayerWeights.clear()
		self.tabLayerOutput.clear()
		self.train_valid_accuracy.update_figure([1], [1], 0)
		self.train_valid_loss.update_figure([1], [1], 0)
		self.lblConvWeightInfo.setText('Visualizations of the Convolution Layers Weights will appear shortly after the Testing Set has been Evaluated')
		self.lblConvOutputInfo.setText('Visualizations of the Output of Convolution Layers will appear shortly after the Testing Set has been Evaluated')

	# called to delete all layers of new model
	def delete_model_button_clicked(self):
		if not self.new_model:
			self.txtOutputModelLog.append("No Model to Delete!")
			return
		self.new_model = []
		self.layer_names = []
		self.lstModel.clear()

	# called to delete just the last layer in model creation
	def delete_last_layer_button_clicked(self):
		try:
			self.new_model.pop(-1)
			self.layer_names.pop(-1)
			self.lstModel.takeItem(len(self.new_model))
		except:
			self.txtOutputModelLog.append("No More Layers to Delete!")

	# called when a combo box index is changed by user
	# @param bias_init_comboBox = reference to GUI element that states bias initiliazation method
	# @param bias_val_spinner = reference to spinner GUI element to take keep rate value from
	# @param bias_val_label = reference to label GUI element that says which bias value measure is in use
	# @param help_icon = reference to help icon GUI element that gives information about bias value measure
	def weight_init_combo_changed(self, weight_init_combobox, weight_val_spinner, weight_val_label, help_icon):
		# if random or truncated normal is used, show standard deviation option
		if weight_init_combobox.currentIndex() == 0 or weight_init_combobox.currentIndex() == 1:
			weight_val_label.setText('Std Dev:')
			weight_val_spinner.setFixedWidth(60)
			help_icon.setFixedWidth(24)
			# if constant is used, show constant val option
		elif weight_init_combobox.currentIndex() == 2:
			weight_val_label.setText('')
			weight_val_spinner.setFixedWidth(0)
			help_icon.setFixedWidth(0)

	# called when a combo box index is changed by user
	# @param bias_init_comboBox = reference to GUI element that states bias initiliazation method
	# @param bias_val_spinner = reference to spinner GUI element to take keep rate value from
	# @param bias_val_label = reference to label GUI element that says which bias value measure is in use
	# @param help_icon = reference to help icon GUI element that gives information about bias value measure
	def bias_init_combo_changed(self, bias_init_combobox, bias_val_spinner, bias_val_label, help_icon):
		if bias_init_combobox.currentIndex() == 0:
			# if zeros init is used, hide all fields relating to a bias value
			bias_val_label.setText('')
			bias_val_spinner.setFixedWidth(0)
			help_icon.setFixedWidth(0)
			# if random or truncated normal is used, show standard deviation option
		elif bias_init_combobox.currentIndex() == 1 or bias_init_combobox.currentIndex() == 2:
			bias_val_label.setText('Std Dev:')
			bias_val_spinner.setFixedWidth(60)
			help_icon.setFixedWidth(24)
			# if constant is used, show constant val option
		elif bias_init_combobox.currentIndex() == 3:
			bias_val_label.setText('Value:')
			bias_val_spinner.setFixedWidth(60)
			help_icon.setFixedWidth(24)

	# called when the testing/validation split spinner is changed
	def testing_validation_split_spinner_changed(self):
		# autofill value of validation field depending on testing field
		test_split = self.spnTestSplit.text()
		valid_split = 100 - int(test_split)
		self.txtValidSplit.setText(str(valid_split))

	# called whenever a change directory button is pressed
	# @param path_text_field = reference to GUI element that will be filled with new directory
	# @param disable = bool value that states whether to reset statis of save path checkbox
	def change_directory_path(self, path_text_field, disable=False):
		path = QFileDialog.getExistingDirectory(self, "Select Directory")
		path_text_field.setText(path + "/")
		if disable == True:
			self.cbxSavePath.setCheckState(False)

	# called when the user tries to change the model file load path
	def change_file_path(self):
		file = QFileDialog.getOpenFileName(self, 'Open file', '/home', "XML Files (*.xml)")
		self.txtLoadModel.setText(str(file[0])) 

	# used when user toggles between differen data sets
	# @regression = bool if a regression task is selected
	def data_set_rad_state_changed(self, regression):
		# if regression task is selected, hide validation set option and show person selection option
		if regression == True:
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

	# returns string of data set selected depending on which radio button is currently selected
	def current_data_set(self):
		if self.radCIFAR10.isChecked():
			return 'CIFAR10'
		elif self.radMNIST.isChecked():
			return 'MNIST'
		elif self.radPrimaHeadPosePitch.isChecked():
			return 'PrimaHeadPosePitch'
		elif self.radPrimaHeadPoseYaw.isChecked():
			return 'PrimaHeadPoseYaw'

	# hides or shows momentum option if selected in optimizer to be used
	def optimizer_combo_box_changed(self):
		if self.cbxOptimizer.currentIndex() == 4:
			self.spnMomentum.setFixedWidth(50)
			self.spnMomentum.setFixedHeight(20)
			self.lblMomentum.setText('Momentum:')
		else:
			self.spnMomentum.setFixedWidth(0)
			self.lblMomentum.setText('')   

	# enables/disables beta value for L2Reg is selected
	def l2_reg_checkbox_state_changed(self):
		if self.chkL2Reg.isChecked():
			self.spnRegBeta.setEnabled(True)
		else: self.spnRegBeta.setEnabled(False)

	# sets save path the same as load path if checkbox is pressed
	def checkpoint_checkbox_state_changed(self):
		if self.cbxSavePath.isChecked():
			self.txtSavePath.setText(self.txtLoadCheckpoints.text())

	# enables/disables keep rate spinner if dropbox checkbox state changes
	# @param dropbox_checkbox = reference to dropbox GUI element that has changed state
	# @param keep_rate_spinner = reference to spinner GUI element to enable/disable
	def dropout_checkbox_state_changed(self, dropout_checkbox, keep_rate_spinner):
		if dropout_checkbox.isChecked():
			keep_rate_spinner.setEnabled(True)
		else: keep_rate_spinner.setEnabled(False)

	# called when user presses create model radio button, takes to design page
	# @param enabled = bool value if radio button is enabled, disables it
	def create_model_rad_clicked(self, enabled):
		if enabled:
			self.tabPages.setCurrentIndex(1)
			self.radLoadModel.setChecked(True)
			
	# called to open various tags,
	# @param index = index of tab to open
	def open_tab(self, index):
		self.tabPages.setCurrentIndex(index)

	# user to clear the Train output log
	def clear_output_log(self):
		self.txtOutputLog.setText('')

	# user to clear the model output log
	def clear_output_model_log(self):
		self.txtOutputModelLog.setText('')

	# called when user presses close application icon, makes them confirm
	def close_event(self):
		reply = QMessageBox.question(self, 'Message', "Are you sure you want to quit? All Unsaved Progress will be lost...", 
			QMessageBox.Yes, QMessageBox.No)
		if reply == QMessageBox.Yes:
			self.close()

	# called to update the progress of training at each epoch
	# @param progress = float value containing epoch number
	@pyqtSlot(float)
	def update_progress_bar(self, progress: float):
		self.prgTrainingProgress.setValue(progress)

	# called when a computational model is loaded, prints details to log
	# @param layers = list of all layers loaded from .xml
	@pyqtSlot(list)
	def show_model_details(self, layers: list):
		for e in layers:
			self.txtOutputLog.append("Layer Name: {0}".format(e.layer_name))
			if e.layer_type == 'Convolution':
				self.txtOutputLog.append('Type: Convolution')
				self.txtOutputLog.append("Num Output Filters {0}".format(e.num_output_filters))
				self.txtOutputLog.append("Kernel Size: [1,{0},{1},1], Stride: [1,{2},{3},1]".format(e.kernel_size, e.kernel_size, e.stride, e.stride))
				self.txtOutputLog.append("Activation Function: {0}".format(e.act_function))  
				if e.weight_init == 'Xavier':
					self.txtOutputLog.append("Weight Initialization: {0}".format(e.weight_init))
				else: self.txtOutputLog.append("Weight Initialization: {0}, Standard Deviation: {1}".format(e.weight_init, e.weight_val))
				if e.bias_init == 'Random Normal' or e.bias_init == 'Truncated Normal':
					self.txtOutputLog.append("Bias Initialization: {0}, Standard Deviation: {1}".format(e.bias_init, e.bias_val))
				elif e.bias_init == 'Constant':
					self.txtOutputLog.append("Bias Initialization: {0}, Value: {1}".format(e.bias_init, e.bias_val))
				else: self.txtOutputLog.append("Bias Initialization: {0}".format(e.bias_init))
				self.txtOutputLog.append("Padding: {0}".format(e.padding))  
				self.txtOutputLog.append("Normalize: {0}".format(e.normalize))  
				self.txtOutputLog.append("Dropout: {0}, Keep Rate {1} \n".format(e.dropout, e.keep_rate)) 
			elif e.layer_type == 'Max Pool':
				self.txtOutputLog.append('Type: Max Pooling')
				self.txtOutputLog.append("Kernel Size: [1,{0},{1},1], Stride: [1,{2},{3},1]".format(e.kernel_size, e.kernel_size, e.stride, e.stride))
				self.txtOutputLog.append("Padding: {0}".format(e.padding))  
				self.txtOutputLog.append("Normalize: {0}".format(e.normalize))  
				self.txtOutputLog.append("Dropout: {0}, Keep Rate {1} \n".format(e.dropout, e.keep_rate)) 
			elif e.layer_type == 'Fully Connected':
				self.txtOutputLog.append('Type: Fully Connected')
				self.txtOutputLog.append("Num Output Nodes {0}".format(e.num_output_nodes))
				self.txtOutputLog.append("Activation Function: {0}".format(e.act_function))
				if e.weight_init == 'Xavier':
					self.txtOutputLog.append("Weight Initialization: {0}".format(e.weight_init))
				else: self.txtOutputLog.append("Weight Initialization: {0}, Standard Deviation: {1}".format(e.weight_init, e.weight_val))  
				if e.bias_init == 'Random Normal' or e.bias_init == 'Truncated Normal':
					self.txtOutputLog.append("Bias Initialization: {0}, Standard Deviation: {1}".format(e.bias_init, e.bias_val))
				elif e.bias_init == 'Constant':
					self.txtOutputLog.append("Bias Initialization: {0}, Value: {1}".format(e.bias_init, e.bias_val))
				else: self.txtOutputLog.append("Bias Initialization: {0}".format(e.bias_init))
				self.txtOutputLog.append("Dropout: {0}, Keep Rate {1} \n".format(e.dropout, e.keep_rate)) 
			elif e.layer_type == 'Output':
				self.txtOutputLog.append('Type: Output')
				self.txtOutputLog.append("Activation Function: {0}".format(e.act_function))  
				if e.weight_init == 'Xavier':
					self.txtOutputLog.append("Weight Initialization: {0}".format(e.weight_init))
				else: self.txtOutputLog.append("Weight Initialization: {0}, Standard Deviation: {1}".format(e.weight_init, e.weight_val))
				if e.bias_init == 'Random Normal' or e.bias_init == 'Truncated Normal':
					self.txtOutputLog.append("Bias Initialization: {0}, Standard Deviation: {1}".format(e.bias_init, e.bias_val))
				elif e.bias_init == 'Constant':
					self.txtOutputLog.append("Bias Initialization: {0}, Value: {1}".format(e.bias_init, e.bias_val))
				else: self.txtOutputLog.append("Bias Initialization: {0}".format(e.bias_init))
		self.txtOutputLog.append('\n------------------------------------------------------------------------------------------------------------- \n')

	# called when switching between regression and classification tasks, changes title of accuracy grapg
	@pyqtSlot(bool)
	def update_graph_titles(self, regression: bool):
		if regression == True:
			self.lblTrainValidAcc.setText('Mean Absolute Error over Epochs')
			self.lblTrainValidLoss.setText('Root Mean Squared Error over Epochs')
			self.lblLossLegend.setFixedWidth(0)
			self.lblAccuracyLegend.setFixedWidth(0)
			self.tabVisualizations.setTabText(2, 'Batch MAE / RMSE')
		else:
			self.lblTrainValidLoss.setText('Training/Validation Loss Over Epochs')
			self.lblTrainValidAcc.setText('Training/Validation Accuracy Over Epochs')
			self.lblLossLegend.setFixedWidth(200)
			self.lblAccuracyLegend.setFixedWidth(200)
			self.tabVisualizations.setTabText(2, 'Training / Validation Accuracy')


	# called when CIFAR10 is run, shows the key below confusion plots
	@pyqtSlot(bool)
	def show_classification_key(self, regression: bool):
		self.grpClassKey.setFixedWidth(1100)
		if regression == True:
			self.lblClassKey.setText('Key for Prima Classes (Range in degrees)')
			self.lblClasses.setText(' 0: [-90, -75]   1: [-75, -60]   2: [-60, -45]   3: [-45, -30]   4: [-30, -15]   5: [-15, 0]   6: [0, 15]   7: [15, 30]   8: [30, 45]   9: [45, 60]   10: [60, 75]   11: [75, 90]')
		else:
			self.lblClassKey.setText('Key for CIFAR-10 Classes')
			self.lblClasses.setText('0 = Airplane         1 = Automobile         2 = Bird         3 = Cat         4 = Deer         5 = Dog         6 = Frog         7 = Horse         8 = Ship         9 = Truck')

	# signals the GUI to load convolution layer weight visualizations from disk and embed
	# @param weight_file_names = list containing file paths of all conv layer weight visualizations
	# @param conv_layer_names = list containing file paths of all conv layer names
	@pyqtSlot(list, list)
	def embed_network_weights(self, weight_file_names: list, conv_layer_names: list):
		try:
			self.lblConvWeightInfo.setText('Note: Positive weights are shown in Red, Negative weights are displayed as Blue, White signify weights around Zero.')

			count = 0
			# create tab for each file name/visualization present
			for file_name in weight_file_names:
				self.tab = QWidget()
				tab_name = conv_layer_names[count]
				self.tabLayerWeights.addTab(self.tab, tab_name)
				self.image = QLabel()
				self.vbox = QVBoxLayout()
				self.vbox.addWidget(self.image)
				# load visualization from file pathway and embed into newly created tab
				pix_map = QPixmap(file_name)
				self.image.setPixmap(pix_map)
				self.tab.setLayout(self.vbox)
				count += 1
		except AttributeError:
			return 0

	# signals the GUI to load convolution output visualizations from disk and embed
	# @param output_file_names = list containing file paths of all conv output visualizations
	# @param conv_layer_names = list containing file paths of all conv layer names
	@pyqtSlot(list, list)
	def embed_network_outputs(self, output_file_names: list, conv_layer_names: list):
		try:
			self.lblConvOutputInfo.setText('Note: Each image represents one of the maps created when passing the corresponding Convolution Kernel across the image.')

			count = 0
			# create tab for each file name/visualization present
			for file_name in output_file_names:
				self.tab = QWidget()
				tab_name = conv_layer_names[count]
				self.tabLayerOutput.addTab(self.tab, tab_name)
				self.image = QLabel()
				self.vbox = QVBoxLayout()
				self.vbox.addWidget(self.image)
				# load visualization from file pathway and embed into newly created tab
				pix_map = QPixmap(file_name)
				self.image.setPixmap(pix_map)
				self.tab.setLayout(self.vbox)	
				count += 1
		except AttributeError:
			return 0

	# signals the GUI to load sample images and predictions visualizations from disk and embed
	# @param file_name = string containing location of confusion plot to load
	# @param with_predictions = bool if plot is for predictions
	@pyqtSlot(str, bool)
	def embed_sample_images(self, file_name: str, with_predictions: bool):
		# load image from file pathway
		pix_map = QPixmap(file_name)

		if with_predictions == True:
			self.lblSamplePredictions.setPixmap(pix_map)
		else:
			self.lblSampleImages.setPixmap(pix_map)


	# this function loads the created confusion matrix from disk and loads into GUI
	# @param file_name = string containing location of confusion plot to load
	# @param epoch = int value of current epoch
	# @param training = bool if this confusion plot if for training batch or testing set
	@pyqtSlot(str, int, bool)
	def update_confusion_plot(self, file_name: str, epoch: int, training: bool):
		
		pix_map = QPixmap(file_name)

		if training == True:
			self.lblTrainingConfusionMat.setPixmap(pix_map)
			self.lblBatchConf.setText("Training Batch {0}".format(epoch))
		else:
			self.lblTestingConfusionMat.setPixmap(pix_map)


	# called when thread(s) have finished, i.e. training has finished or been cancelled
	@pyqtSlot()
	def abort_workers(self):
		# cycle all threads and signal to quit work
		for thread, worker in self.__threads: 
			thread.quit() 
			thread.wait() 
		self.btnTrainNetwork.setEnabled(True)
	
	# function that signals to the worked thread that the user has requested training to stop
	def cancel_train(self):
		self.end_train.emit()

		
if __name__ == "__main__":
	app = QApplication(sys.argv)
	ui = CNNApp()
	ui.show()
	sys.exit(app.exec_())
