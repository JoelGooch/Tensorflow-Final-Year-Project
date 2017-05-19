import tensorflow as tf
import numpy as np
import os
import pickle
import datetime
import sys
import cv2



# This script was used for testing the combined network, *AWFL and McGill of Training, AFW for Testing*
# it combines the data sets and follows a standard procedure.


# this function calculates the accuracy measurements from the predicted labels and actual labels
# it also writes results to a .csv file if requested
# 	@param predictions = numpy array containing predicted values by network
# 	@param labels = numpy array containing actual values 
# 	@param test_or_train = string containing 'Training' or 'Testing', used for .csv file name
# 	@param write_to_csv = boolean value that states whether the user wants a .csv file of results to be written 
def evaluate_accuracy(predictions, labels, test_or_train, test, write_to_csv=False):

	# Convert prediction to degrees
	prediction_degree = predictions * 90
	actual_degree = labels * 90

	# calculate Root Mean Squared Error 
	RMSE = np.sqrt(np.sum(np.square(prediction_degree - actual_degree), dtype=np.float32) * 1 / predictions.shape[0])
	# calculate Standard Deviation of Root Mean Squared Error 
	RMSE_stdev = np.std(np.sqrt(np.square(prediction_degree - actual_degree)), dtype=np.float32)

	# calculate Mean Absolute Error
	MAE = np.sum(np.absolute(prediction_degree - actual_degree), dtype=np.float32) * 1 / predictions.shape[0]
	# calculate Standard Deviation of Mean Absolute Error
	MAE_stdev = np.std(np.absolute(prediction_degree - actual_degree), dtype=np.float32)

	# if user wants results to be written to .csv file
	if write_to_csv == True:

		num_rows, _ = labels.shape
		now = datetime.datetime.now()

		# open (or create) .csv file and write accuracy details within
		file = open('/tmp/COMBINED/Yaw/Test_' + str(test) + '/log/' + test_or_train + '_yaw' + '.csv', 'a')
		file.write('RMSE = ' + str(RMSE) + ',RMSE Standard Deviation = ' + str(RMSE_stdev) + ',MAE = ' + str(MAE) + ',MAE Standard Deviation = ' + str(MAE_stdev) + '\n\n')
		file.write('Predicted Degree,Actual Degree,Error \n')

		for i in range(0, num_rows):
			error = (prediction_degree[i] - actual_degree[i])
			file.write(str(prediction_degree[i]) + ',' + str(actual_degree[i]) + ',' + str(error) + '\n')
			
		file.close()

	return RMSE, RMSE_stdev, MAE, MAE_stdev


# simple helper function that prints accuracy details to console
#	@param data_set = string that appends to front of results to say where they came from
#	@param RMSE = Root Mean Squared Error value 
#	@param RMSE_stdev = Standard Deviation of Root Mean Squared Error value 
#	@param MAE = Mean Absolute Error value
#	@param MAE_stdev = Standard Deviation of Mean Absolute Error
def print_to_console(data_set, RMSE, RMSE_stdev, MAE, MAE_stdev):
	print(data_set + ' RMSE mean: {0} degrees'.format(RMSE))
	print(data_set + ' RMSE std: {0} degrees'.format(RMSE_stdev))
	print(data_set + ' MAE mean: {0} degrees'.format(MAE))
	print(data_set + ' MAE std: {0} degrees \n'.format(MAE_stdev))


def main():

	# perform the entire test 6 times
	for test in range (1, 6):

		# variable params
		total_epochs = 30000
		batch_size = 64
		learning_rate = 0.0005
		trainingdropout_keep_rate = 0.5

		# static params
		image_size = 64 # images are 64x64x3
		num_channels = 3 # RGB
		num_classes = 1 # regression

		########################################################################################################################

		mcgill_pickle_directory = 'C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/data/McGill/Pickle/McGill.pickle'


		with open(mcgill_pickle_directory, mode='rb') as file:
			data = pickle.load(file, encoding='bytes')
			mcgill_data = data['data']
			mcgill_labels = data['labels']
			# garbage collect unused data
			del data
		
		# reshape data 
		mcgill_data = mcgill_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
		mcgill_labels = mcgill_labels.reshape((-1, 1)).astype(np.float32)

		print('\nmcgill data shape {0}'.format(mcgill_data.shape))
		print('mcgill labels shape {0}\n'.format(mcgill_labels.shape))


		# normalize data
		mcgill_data /= 255 
		mcgill_labels /= 100

		########################################################################################################################

		AFLW_pickle_directory = 'C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/data/AFLW/afwl_f1.pickle'

		with open(AFLW_pickle_directory, mode='rb') as file:
			data = pickle.load(file, encoding='bytes')
			AFLW_data = data[b'training_dataset']
			AFLW_labels = data[b'training_label_yaw']
			AFLW_test_data = data[b'test_dataset']
			AFLW_testing_labels = data[b'test_label_yaw']
			# garbage collect unused data
			del data
		
		# reshape data 
		AFLW_data = AFLW_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
		AFLW_labels = AFLW_labels.reshape((-1, 1)).astype(np.float32)
		AFLW_test_data = AFLW_test_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
		AFLW_testing_labels = AFLW_testing_labels.reshape((-1, 1)).astype(np.float32)

		print('AFLW training data shape BEFORE join {0}'.format(AFLW_data.shape))
		print('AFLW training labels shape BEFORE join {0}'.format(AFLW_labels.shape))
		print('AFLW testing data shape BEFORE join {0}'.format(AFLW_test_data.shape))
		print('AFLW testing labels shape BEFORE join {0} \n'.format(AFLW_testing_labels.shape))

		AFLW_data = np.append(AFLW_data, AFLW_test_data)
		AFLW_labels = np.append(AFLW_labels, AFLW_testing_labels)

		AFLW_data = AFLW_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
		AFLW_labels = AFLW_labels.reshape((-1, 1)).astype(np.float32)


		print('AFLW training data shape AFTER join {0}'.format(AFLW_data.shape))
		print('AFLW training labels shape AFTER join {0} \n'.format(AFLW_labels.shape))

		# bring AFLW labels from [-100, 100] range to [-90, 90]
		#AFLW_labels *= 100
		# AFLW labels are reverse in comparison to McGill and AFW
		AFLW_labels *= -1

		#np.interp(AFLW_labels, [-100, 100], [-90, 90])
		#AFLW_labels /= 100

		# normalize data
		AFLW_data -= 127 

		del AFLW_test_data
		del AFLW_testing_labels

		########################################################################################################################


		training_data = np.append(AFLW_data, mcgill_data)
		training_labels = np.append(AFLW_labels, mcgill_labels)

		training_data = training_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
		training_labels = training_labels.reshape((-1, 1)).astype(np.float32)

		print('combined training data shape AFTER join {0}'.format(training_data.shape))
		print('combined training labels shape AFTER join {0} \n'.format(training_labels.shape))

		del AFLW_data
		del AFLW_labels
		del mcgill_data
		del mcgill_labels
 

		########################################################################################################################


		AFW_pickle_directory = 'C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/data/AFW/Yaw/afw_f1.pickle'
		
		with open(AFW_pickle_directory, mode='rb') as file:
			data = pickle.load(file, encoding='bytes')
			AFW_data = data[b'training_dataset']
			AFW_labels = data[b'training_label_yaw']
			AFW_test_data = data[b'test_dataset']
			AFW_testing_labels = data[b'test_label_yaw']
			# garbage collect unused data
			del data

		# reshape data 
		AFW_data = AFW_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
		AFW_labels = AFW_labels.reshape((-1, 1)).astype(np.float32)
		AFW_test_data = AFW_test_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
		AFW_testing_labels = AFW_testing_labels.reshape((-1, 1)).astype(np.float32)

		print('AFW training data shape BEFORE join {0}'.format(AFW_data.shape))
		print('AFW training labels shape BEFORE join {0}'.format(AFW_labels.shape))
		print('AFW testing data shape BEFORE join {0}'.format(AFW_test_data.shape))
		print('AFW testing labels shape BEFORE join {0} \n'.format(AFW_testing_labels.shape))

		testing_data = np.append(AFW_data, AFW_test_data)
		testing_labels = np.append(AFW_labels, AFW_testing_labels)

		testing_data = testing_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
		testing_labels = testing_labels.reshape((-1, 1)).astype(np.float32)

		testing_data -= 127

		# bring labels in range -1 to 1
		testing_labels /= 90

		print('testing data shape AFTER join {0}'.format(testing_data.shape))
		print('testing labels shape AFTER join {0}\n'.format(testing_labels.shape))


		########################################################################################################################


		# create and define Tensorflow computational graph
		graph = tf.Graph()
		with graph.as_default():
			
			 # print some example images and labels to check all is in order
			'''
			for i in range(0, 500):
				# select and print random image from testing set 
				random = np.random.randint(0, training_data.shape[0])
				img = np.copy(training_data[random] * 255)
				test_class = training_labels[random] 
				#img = img.astype(np.uint8)
				#img = img.reshape(64, 64, num_channels)
				cv2.imshow('image', img)
				print("image class = " + str(test_class))
				cv2.waitKey(0)
				cv2.destroyAllWindows()
			'''
		 

			# define placeholder variables
			x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))    # for input data
			y = tf.placeholder(tf.float32, shape=(None, num_classes)) 							  # for input labels
			keep_rate = tf.placeholder(tf.float32) 	

			#					[kernel size, kernel size, num input channels, num output channels]
			#conv_1_weights = tf.Variable(tf.truncated_normal([3, 3, num_channels, 64], stddev=0.1), name="conv1y_w")
			conv_1_weights = tf.get_variable(name="conv1y_w", shape=[3, 3, num_channels, 64], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			conv_1_biases = tf.Variable(tf.zeros([64]), name="conv1y_b")

			#conv_2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), name="conv2y_w")
			conv_2_weights = tf.get_variable(name="conv2y_w", shape=[3, 3, 64, 128], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			conv_2_biases = tf.Variable(tf.zeros([128]), name="conv2y_b")

			#conv_3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1), name="conv3y_w")
			conv_3_weights = tf.get_variable(name="conv3y_w", shape=[3, 3, 128, 256], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			conv_3_biases = tf.Variable(tf.zeros([256]), name="conv3y_b")

			#conv_3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1), name="conv3y_w")
			conv_4_weights = tf.get_variable(name="conv4y_w", shape=[3, 3, 256, 512], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			conv_4_biases = tf.Variable(tf.zeros([512]), name="conv4y_b")

			#fully_conn_1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 256, 512], stddev=0.1),name="dense1y_w") 
			fully_conn_1_weights = tf.get_variable(name="dense1p_w", shape=[8 * 8 * 512, 1024], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			fully_conn_1_biases = tf.Variable(tf.zeros([1024]), name="dense1y_b")

			#fully_conn_2_weights = tf.Variable(tf.truncated_normal([512, 512], stddev=0.1),name="dense2y_w") 
			fully_conn_2_weights = tf.get_variable(name="dense2y_w", shape=[1024, 512], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			fully_conn_2_biases = tf.Variable(tf.zeros([512]), name="dense2y_b")

			#output_weights = tf.Variable(tf.truncated_normal([512, num_classes], stddev=0.1), name="outy_w")
			output_weights = tf.get_variable(name="outy_w", shape=[512, num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			output_biases = tf.Variable(tf.zeros([num_classes]), name="outp_b")



			# define network architecture
			#	@param data = data to be fed through network
			#	@param dropout = keep probability for dropout operator, defaults to 1.0 (no dropout chance) if no value supplied
			def CNN_Model(data, dropout=1.0):

				# reshape input data 
				X = tf.reshape(data, shape=[-1, image_size, image_size, num_channels])

				# convolution layer                                 kernel stride = 1   padded with zeros
				conv_1 = tf.tanh(tf.nn.conv2d(X, conv_1_weights, strides=[1, 1, 1, 1], padding='SAME') + conv_1_biases)

				# max pooling layer                kernel size = 2     kernel stride = 2   padded with zeros
				max_pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

				# normalize
				norm_1 = tf.nn.lrn(max_pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

				# dropout layer
				norm1 = tf.nn.dropout(norm_1, dropout)

				# convolution layer                                       kernel stride = 1   padded with zeros
				conv_2 = tf.tanh(tf.nn.conv2d(norm_1, conv_2_weights, strides=[1, 1, 1, 1], padding='SAME') + conv_2_biases)

				# max pooling layer                 kernel size = 2     kernel stride = 2   padded with zeros
				max_pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

				# normalize
				norm_2 = tf.nn.lrn(max_pool_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

				# dropout layer
				norm_2 = tf.nn.dropout(norm_2, dropout)
				
				# convolution layer                                       kernel stride = 1   padded with zeros
				conv_3 = tf.tanh(tf.nn.conv2d(norm_2, conv_3_weights, strides=[1, 1, 1, 1], padding='SAME') + conv_3_biases)

				# max pooling layer                 kernel size = 2     kernel stride = 2   padded with zeros
				max_pool_3 = tf.nn.max_pool(conv_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

				# normalize
				norm_3 = tf.nn.lrn(max_pool_3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

				# dropout layer
				norm_3 = tf.nn.dropout(norm_3, dropout)

				conv_4 =  tf.tanh(tf.nn.conv2d(norm_3, conv_4_weights, strides=[1, 1, 1, 1], padding='SAME') + conv_4_biases)

				# normalize
				norm_4 = tf.nn.lrn(conv_4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

				# dropout layer
				norm_4 = tf.nn.dropout(norm_4, dropout)

				# reshape convolution layer to feed into fully connected
				layer_shape = norm_4.get_shape()
				# extract number of features from shape
				num_features = layer_shape[1:4].num_elements()
				# reshape layer using number of features
				reshaped = tf.reshape(norm_4, [-1, num_features])

				# fully connected layer
				fully_conn_1 = tf.tanh(tf.matmul(reshaped, fully_conn_1_weights) + fully_conn_1_biases) 

				# dropout layer
				fully_conn_1 = tf.nn.dropout(fully_conn_1, dropout)

				# fully connected layer
				fully_conn_2 = tf.tanh(tf.matmul(fully_conn_1, fully_conn_2_weights) + fully_conn_2_biases) 

				# dropout layer
				fully_conn_2 = tf.nn.dropout(fully_conn_2, dropout)

				# Output layer
				output = tf.tanh(tf.matmul(fully_conn_2, output_weights) + output_biases)

				return output
				

			# calculate predictions
			model_output = CNN_Model(x, keep_rate)
			
			# calculate loss
			loss = tf.nn.l2_loss(model_output - y)
			
			# add L2 regularization term on the weights
			beta = 0.0005
			for e in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
				# if a weights variable (excluding biases), add regularization term
				if 'weights' in e.name:
					loss += (beta * tf.nn.l2_loss(e))

			loss_summary = tf.summary.scalar('loss', loss)

			# variable to keep track of epochs passed across all runs
			global_step = tf.Variable(0, trainable=False)

			# options for Tensorlow optimizers
			#optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(loss, global_step=global_step)
			#optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
			#optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
			#optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95).minimize(loss, global_step=global_step)
			optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp').minimize(loss, global_step=global_step)

			# create saver object to save network to disk
			saver = tf.train.Saver()

			# start Tensorflow session
			with tf.Session(graph=graph) as session:
				# where all summaries will be stored
				merged = tf.summary.merge_all()
				curr_time = datetime.datetime.now()

				log_path = '/tmp/COMBINED/Yaw/Test_' + str(test) + '/log/' + str(curr_time.hour) + str(curr_time.minute) + str(curr_time.second)
				# if folder does not exist, create it
				if not os.path.exists(log_path):
					os.makedirs(log_path)

				summaries = tf.summary.FileWriter(log_path, graph)

				save_path = '/tmp/COMBINED/Yaw/checkpoints/Test_' + str(test)
				# if folder does not exist, create it
				if not os.path.exists(save_path):
					os.makedirs(save_path)

				'''
				# try to load previous checkpoint, if it doesnt exist, start from fresh
				try:

					print('Trying to restore last checkpoint ...')
					# Use TensorFlow to find the latest checkpoint - if any.
					last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)

					# Try and load the data in the checkpoint.
					saver.restore(session, save_path=last_chk_path)
					print('Restored checkpoint from: {0}'.format(last_chk_path))
				except:
					print('Failed to restore checkpoint. Initializing variables instead.')
					session.run(tf.global_variables_initializer())
				'''

				session.run(tf.global_variables_initializer())

				# here I create and format the .csv file that will hold all the information at each batch
				file = open('/tmp/COMBINED/Yaw/Test_' + str(test) + '/log/batchstats_yaw.csv', 'a')
				file.write(',B Loss,B RMSE,B RMSE StDev,B MAE,B MAE StDev,,V RMSE,V RMSE StDev,V MAE,V MAE StDev\n')
				file.close()

				# start training loop, the + 1 ensures that results are printed on the final epoch
				for epoch in range(total_epochs + 1):

					# take random batch from whole training set
					rand_num = np.random.randint(0, 27353, (64))
					batch_data = np.take(training_data, rand_num, axis=0, out=None)
					batch_labels = np.take(training_labels, rand_num, axis=0, out=None)

					# fill tensorflow placeholders
					feed_dict = {x: batch_data, y: batch_labels, keep_rate: 0.5}

					# run tensorflow conputation graph
					_, batch_loss, batch_predictions, batch_summary = session.run([optimizer, loss, model_output, merged], feed_dict=feed_dict)

					# add summary at epoch for tensorboard
					summaries.add_summary(batch_summary, epoch)

					# print batch information and evaluate validation accuracy every 100 epochs
					if (epoch % 100 == 0):
						print('--------------------------------------------------------------------')
						print('Global Step: {0} \n'.format(str(global_step.eval())))
						print('Loss at epoch: {0} of {1} is {2} \n'.format(epoch, str(total_epochs), batch_loss))

						''' # print some example tests
						print('predicted {0}, actual {1}'.format(batch_predictions[0],batch_labels[0]))
						print('predicted {0}, actual {1}'.format(batch_predictions[10],batch_labels[10]))
						print('predicted {0}, actual {1}'.format(batch_predictions[11],batch_labels[11]))
						print('predicted {0}, actual {1}'.format(batch_predictions[13],batch_labels[13]))
						print('predicted {0}, actual {1}'.format(batch_predictions[15],batch_labels[15]))
						'''
						
						# evaluate batch accuracy and print to console
						batch_RMSE, batch_RMSE_stdev, batch_MAE, batch_MAE_stdev = evaluate_accuracy(batch_predictions, batch_labels, 'Training', test, False)
						print_to_console('Batch', batch_RMSE, batch_RMSE_stdev, batch_MAE, batch_MAE_stdev)

						
						# run validation set (test set) through network 
						feed_dict = {x: testing_data, y: testing_labels, keep_rate: 1.0}
						validation_predictions = session.run(model_output, feed_dict=feed_dict)

						# evaluate validation accuracy and print to console
						validation_RMSE, validation_RMSE_stdev, validation_MAE, validation_MAE_stdev = evaluate_accuracy(validation_predictions, testing_labels, 'Testing', test, False)
						print_to_console('Validation', validation_RMSE, validation_RMSE_stdev, validation_MAE, validation_MAE_stdev)

						# now write all the batch information into the previously made .csv file
						file = open('/tmp/COMBINED/Yaw/Test_' + str(test) + '/log/batchstats_yaw.csv', 'a')
						file.write('Batch ' + str(epoch) + ',' + str(batch_loss) + ',' + str(batch_RMSE) + ',' + str(batch_RMSE_stdev) + ',' + str(batch_MAE) + ',' + 
							str(batch_MAE_stdev) + ',,' + str(validation_RMSE) + ',' + str(validation_RMSE_stdev) + ',' + str(validation_MAE) + ',' + str(validation_MAE_stdev)  + '\n')
						file.close()
						
						

					# save state every 10000 epochs in case of power failure or other unexpected event
					# the > 0 part stops it from saving a checkpoint at epoch 0
					#if (epoch % 10000 == 0 and epoch > 0):
					#	saver.save(session, save_path=save_path, global_step=global_step)
					#	print('Saved Checkpoint')
						

				# then run testing set through the network and evaluate accuracy
				feed_dict = {x: testing_data, y: testing_labels, keep_rate: 1.0}
				test_set_predictions = session.run(model_output, feed_dict=feed_dict)
				test_RMSE, test_RMSE_stdev, test_MAE, test_MAE_stdev = evaluate_accuracy(test_set_predictions, testing_labels, 'Testing', test, True)
				
				print('--------------------------------------------------------------------')
				print_to_console('Test Set', test_RMSE, test_RMSE_stdev, test_MAE, test_MAE_stdev)
				print('--------------------------------------------------------------------')

				saver.save(session, save_path=save_path, global_step=global_step)
				print('Saved Checkpoint')

			
if __name__ == '__main__':
	main()