import tensorflow as tf
import numpy as np
import os
import pickle
import datetime
import sys
import cv2 as cv

# this function calculates the accuracy measurements from the predicted labels and actual labels
# it also writes results to a .csv file if requested
# 	@param predictions = numpy array containing predicted values by network
# 	@param labels = numpy array containing actual values 
# 	@param pickle_file = integer value of which person is being used for the testing set
# 	@param test_or_train = string containing 'Training' or 'Testing', used for .csv file name
# 	@param write_to_csv = boolean value that states whether the user wants a .csv file of results to be written 
def evaluate_accuracy(predictions, labels, pickle_file, test_or_train, write_to_csv=False):

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
	if write_to_csv == True:

		num_rows, _ = labels.shape
		now = datetime.datetime.now()

		# open (or create) .csv file and write accuracy details within
		file = open('/tmp/Prima/Yaw/log/' + test_or_train + '_person' + str(pickle_file) + '_yaw' + str(now.hour) + str(now.minute) + str(now.second) + '.csv', 'a')
		file.write('Evaluating ' + test_or_train + ' Set, Using Person ' + str(pickle_file) + ' as Test Set\n\n')
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

	# variable params
	total_epochs = 0
	batch_size = 64
	learning_rate = 0.005
	trainingdropout_keep_rate = 0.5

	# static params
	image_size = 64 # images are 64x64x3
	num_channels = 3 # RGB
	num_classes = 1 # regression
	pickle_directory = 'C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/data/Prima Head Pose/yaw/prima_yaw_p'

	# there are 15 files (15 people), each uses jack knife cross validation meaning in each file a different person is used for the testing set
	# essentially this makes 15 separate networks, the final accuracy terms are the mean across all networks
	for pickle_file in range(1, 16):

		print('\nRunning with Person {0} as the testing set\n'.format(str(pickle_file)))

		current_pickle = pickle_directory + str(pickle_file) + '_out.pickle'

		with open(current_pickle, mode='rb') as file:
			data = pickle.load(file, encoding='bytes')
			training_data = data[b'training_dataset']
			training_labels = data[b'training_label']
			testing_data = data[b'validation_dataset']
			testing_labels = data[b'validation_label']
			# garbage collect unused data
			del data
		
		# reshape data 
		training_data = training_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
		training_labels = training_labels.reshape((-1, 1)).astype(np.float32)
		testing_data = testing_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
		testing_labels = testing_labels.reshape((-1, 1)).astype(np.float32)


		# normalize data
		training_data -= 127 
		testing_data -= 127

		cv.imshow('image', training_data[10])
		cv.waitKey(0)
		cv.destroyAllWindows()

		# create and define Tensorflow computational graph
		graph = tf.Graph()
		with graph.as_default():

			# define placeholder variables
			x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))    # for input data
			y = tf.placeholder(tf.float32, shape=(None, num_classes)) 							  # for input labels
			keep_rate = tf.placeholder(tf.float32) 												  # keep probabilty for dropout

			# define network weights and biases
			# [kernel size, kernel size, num input channels, num output channels] 
			conv_1_weights = tf.Variable(tf.truncated_normal([3, 3, num_channels, 64], stddev=0.1), name="conv_1_weights")
			conv_1_biases = tf.Variable(tf.zeros([64]), name="conv_1_biases")

			conv_2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), name="conv_2_weights")
			conv_2_biases = tf.Variable(tf.zeros([128]), name="conv_2_biases")

			# 16 * 16 because of 2 max pooling layers with stride = 2 : 64 -> 32 -> 16
			fully_conn_1_weights = tf.Variable(tf.truncated_normal([16 * 16 * 128, 128], stddev=0.1),name="fully_conn_1_weights") 
			fully_conn_1_biases = tf.Variable(tf.zeros([128]), name="fully_conn_1_biases")

			output_weights = tf.Variable(tf.truncated_normal([128, num_classes], stddev=0.1), name="output_weights")
			output_biases = tf.Variable(tf.zeros([num_classes]), name="output_biases")

			# define network architecture
			#	@param data = data to be fed through network
			#	@param dropout = keep probability for dropout operator, defaults to 1.0 (no dropout chance) if no value supplied
			def CNN_Model(data, dropout=1.0):

				# reshape input data 
				X = tf.reshape(data, shape=[-1, image_size, image_size, num_channels])

				# convolution layer                                 kernel stride = 1   padded with zeros
				conv_1 = tf.sigmoid(tf.nn.conv2d(X, conv_1_weights, strides=[1, 1, 1, 1], padding='SAME') + conv_1_biases)

				# max pooling layer                 kernel size = 2     kernel stride = 2   padded with zeros
				max_pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

				# normalize
				norm_1 = tf.nn.lrn(max_pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

				# dropout layer
				norm1 = tf.nn.dropout(norm_1, dropout)

				# convolution layer                                       kernel stride = 1   padded with zeros
				conv_2 = tf.sigmoid(tf.nn.conv2d(norm_1, conv_2_weights, strides=[1, 1, 1, 1], padding='SAME') + conv_2_biases)

				# max pooling layer                 kernel size = 2     kernel stride = 2   padded with zeros
				max_pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

				# normalize
				norm_2 = tf.nn.lrn(max_pool_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

				# dropout layer
				norm_2 = tf.nn.dropout(norm_2, dropout)

				# reshape convolution layer to feed into fully connected
				layer_shape = norm_2.get_shape()
				# extract number of features from shape
				num_features = layer_shape[1:4].num_elements()
				# reshape layer using number of features
				reshaped = tf.reshape(norm_2, [-1, num_features])

				# fully connected layer
				fully_conn_1 = tf.sigmoid(tf.matmul(reshaped, fully_conn_1_weights) + fully_conn_1_biases) 

				# dropout layer
				fully_conn_1 = tf.nn.dropout(fully_conn_1, dropout)

				# Output layer
				output = tf.sigmoid(tf.matmul(fully_conn_1, output_weights) + output_biases)

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

			# options for dynamic learning rates
			#learning_rate = tf.train.exponential_decay(0.0125, global_step, 15000, 0.1, staircase=True)
			#learning_rate = tf.train.exponential_decay(0.001, global_step, 50, 0.65, staircase=True)
			#learning_rate_summary = tf.summary.scalar('learning rate', learning_rate)

			# options for Tensorlow optimizers
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
			optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
			#optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
			#optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95).minimize(loss, global_step=global_step)
			#optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp').minimize(loss, global_step=global_step)

			# create saver object to save network to disk
			saver = tf.train.Saver()

			# start Tensorflow session
			with tf.Session(graph=graph) as session:
				# where all summaries will be stored
				merged = tf.summary.merge_all()
				curr_time = datetime.datetime.now()

				log_path = '/tmp/Prima/Yaw/log/person' + str(pickle_file) + '_' + str(curr_time.hour) + str(curr_time.minute) + str(curr_time.second)
				# if folder does not exist, create it
				if not os.path.exists(log_path):
					os.makedirs(log_path)

				summaries = tf.summary.FileWriter(log_path, graph)

				save_path = '/tmp/Prima/Yaw/checkpoints/person' + str(pickle_file) + '/'
				# if folder does not exist, create it
				if not os.path.exists(save_path):
					os.makedirs(save_path)

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

				# here I create and format the .csv file that will hold all the information at each batch
				file = open('/tmp/Prima/Yaw/log/batchstats_person' + str(pickle_file) + '_yaw.csv', 'a')
				file.write(',B Loss,B RMSE,B RMSE StDev,B MAE,B MAE StDev,,V RMSE,V RMSE StDev,V MAE,V MAE StDev\n')
				file.close()

				# start training loop, the + 1 ensures that results are printed on the final epoch
				for epoch in range(total_epochs + 1):

					# grab random batch from training data
					offset = (epoch * batch_size) % (training_labels.shape[0] - batch_size)
					batch_data = training_data[offset:(offset + batch_size), :, :, :]
					batch_labels = training_labels[offset:(offset + batch_size)]

					# fill tensorflow placeholders
					feed_dict = {x: batch_data, y: batch_labels, keep_rate: trainingdropout_keep_rate}

					# run tensorflow conputation graph
					_, batch_loss, batch_predictions, batch_summary = session.run([optimizer, loss, model_output, merged], feed_dict=feed_dict)

					# add summary at epoch for tensorboard
					summaries.add_summary(batch_summary, epoch)

					# print batch information and evaluate validation accuracy every 100 epochs
					if (epoch % 100 == 0):
						print('--------------------------------------------------------------------')
						print('Global Step: {0} \n'.format(str(global_step.eval())))
						print('Loss at epoch: {0} of {1} is {2} \n'.format(epoch, str(total_epochs), batch_loss))
						#print('Learning Rate: {0}'.format(str(learning_rate.eval())
						
						# evaluate batch accuracy and print to console
						batch_RMSE, batch_RMSE_stdev, batch_MAE, batch_MAE_stdev = evaluate_accuracy(batch_predictions, batch_labels, pickle_file, 'Training', False)
						print_to_console('Batch', batch_RMSE, batch_RMSE_stdev, batch_MAE, batch_MAE_stdev)

						# run validation set (test set) through network 
						feed_dict = {x: testing_data, y: testing_labels, keep_rate: 1.0}
						validation_predictions = session.run(model_output, feed_dict=feed_dict)

						# evaluate validation accuracy and print to console
						validation_RMSE, validation_RMSE_stdev, validation_MAE, validation_MAE_stdev = evaluate_accuracy(validation_predictions, testing_labels, pickle_file, 'Testing', False)
						print_to_console('Validation', validation_RMSE, validation_RMSE_stdev, validation_MAE, validation_MAE_stdev)

						# now write all the batch information into the previously made .csv file
						file = open('/tmp/Prima/Yaw/log/batchstats_person' + str(pickle_file) + '_yaw.csv', 'a')
						file.write('Batch ' + str(epoch) + ',' + str(batch_loss) + ',' + str(batch_RMSE) + ',' + str(batch_RMSE_stdev) + ',' + str(batch_MAE) + ',' + 
							str(batch_MAE_stdev) + ',,' + str(validation_RMSE) + ',' + str(validation_RMSE_stdev) + ',' + str(validation_MAE) + ',' + str(validation_MAE_stdev)  + '\n')
						file.close()
						

					# save state every 5000 epochs in case of power failure or other unexpected event
					# the > 0 part stops it from saving a checkpoint at epoch 0
					if (epoch % 10000 == 0 and epoch > 0):
						saver.save(session, save_path=save_path, global_step=global_step)
						print('Saved Checkpoint')
						

				# when training has stopped run training set through network and evaluate accuracy
				feed_dict = {x: training_data, y:training_labels, keep_rate:1.0}
				training_set_predictions = session.run(model_output, feed_dict=feed_dict)
				train_RMSE, train_RMSE_stdev, train_MAE, train_MAE_stdev = evaluate_accuracy(training_set_predictions, training_labels, pickle_file, 'Training', True)
				
				print('--------------------------------------------------------------------')
				print_to_console('Training Set', train_RMSE, train_RMSE_stdev, train_MAE, train_MAE_stdev)
				print('--------------------------------------------------------------------')

				# then run testing set through the network and evaluate accuracy
				feed_dict = {x: testing_data, y: testing_labels, keep_rate: 1.0}
				test_set_predictions = session.run(model_output, feed_dict=feed_dict)
				test_RMSE, test_RMSE_stdev, test_MAE, test_MAE_stdev = evaluate_accuracy(test_set_predictions, testing_labels, pickle_file, 'Testing', True)
				
				print('--------------------------------------------------------------------')
				print_to_console('Test Set', test_RMSE, test_RMSE_stdev, test_MAE, test_MAE_stdev)
				print('--------------------------------------------------------------------')

				saver.save(session, save_path=save_path, global_step=global_step)
				print('Saved Checkpoint')

			
if __name__ == '__main__':
	main()