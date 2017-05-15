	
  def CNN_Architecture_A():

			# define network weights and biases
			# [kernel size, kernel size, num input channels, num output channels]
			#conv_1_weights = tf.Variable(tf.truncated_normal([3, 3, num_channels, 64], stddev=0.1), name="conv1y_w")
			conv_1_weights = tf.get_variable(name="conv1y_w", shape=[3, 3, num_channels, 64], dtype=tf.float32, initializer=None)
			conv_1_biases = tf.Variable(tf.zeros([64]), name="conv1y_b")

			#conv_2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), name="conv2y_w")
			conv_2_weights = tf.get_variable(name="conv2y_w", shape=[3, 3, 64, 128], dtype=tf.float32, initializer=None)
			conv_2_biases = tf.Variable(tf.zeros([128]), name="conv2y_b")

			# 16 * 16 because of 2 max pooling layers with stride = 2 : 64 -> 32 -> 16
			#fully_conn_1_weights = tf.Variable(tf.truncated_normal([16 * 16 * 128, 128], stddev=0.1),name="dense1p_w")
			fully_conn_1_weights = tf.get_variable(name="dense1p_w", shape=[16 * 16 * 128, 128], dtype=tf.float32, initializer=None)
			fully_conn_1_biases = tf.Variable(tf.zeros([128]), name="dense1y_b")

			#output_weights = tf.Variable(tf.truncated_normal([128, num_classes], stddev=0.1), name="outy_w")
			output_weights = tf.get_variable(name="outy_w", shape=[128, num_classes], dtype=tf.float32, initializer=None)
			output_biases = tf.Variable(tf.zeros([num_classes]), name="outp_b")


				# reshape input data 
				X = tf.reshape(data, shape=[-1, image_size, image_size, num_channels])

				# convolution layer                                 kernel stride = 1   padded with zeros
				conv_1 = tf.tanh(tf.nn.conv2d(X, conv_1_weights, strides=[1, 1, 1, 1], padding='SAME') + conv_1_biases)

				# max pooling layer                 kernel size = 2     kernel stride = 2   padded with zeros
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

				# reshape convolution layer to feed into fully connected
				layer_shape = norm_2.get_shape()
				# extract number of features from shape
				num_features = layer_shape[1:4].num_elements()
				# reshape layer using number of features
				reshaped = tf.reshape(norm_2, [-1, num_features])

				# fully connected layer
				fully_conn_1 = tf.tanh(tf.matmul(reshaped, fully_conn_1_weights) + fully_conn_1_biases) 

				# dropout layer
				fully_conn_1 = tf.nn.dropout(fully_conn_1, dropout)

				# Output layer
				output = tf.tanh(tf.matmul(fully_conn_1, output_weights) + output_biases)

				return output


  def CNN_Architecture_B():

			# define network weights and biases
			# [kernel size, kernel size, num input channels, num output channels] 
			#conv_1_weights = tf.Variable(tf.truncated_normal([3, 3, num_channels, 64], stddev=0.1), name="conv1y_w")
			conv_1_weights = tf.get_variable(name="conv1y_w", shape=[3, 3, num_channels, 64], dtype=tf.float32, initializer=None)
			conv_1_biases = tf.Variable(tf.zeros([64]), name="conv1y_b")

			#conv_2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), name="conv2y_w")
			conv_2_weights = tf.get_variable(name="conv2y_w", shape=[3, 3, 64, 128], dtype=tf.float32, initializer=None)
			conv_2_biases = tf.Variable(tf.zeros([128]), name="conv2y_b")

			#conv_3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1), name="conv3y_w")
			conv_3_weights = tf.get_variable(name="conv3y_w", shape=[3, 3, 128, 256], dtype=tf.float32, initializer=None)
			conv_3_biases = tf.Variable(tf.zeros([256]), name="conv3y_b")

			# 8 * 8 because of 3 max pooling layers with stride = 2 : 64 -> 32 -> 16 -> 8
			#fully_conn_1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 256, 256], stddev=0.1),name="dense1p_w") 
			fully_conn_1_weights = tf.get_variable(name="dense1p_w", shape=[8 * 8 * 256, 256], dtype=tf.float32, initializer=None)
			fully_conn_1_biases = tf.Variable(tf.zeros([256]), name="dense1y_b")

			#output_weights = tf.Variable(tf.truncated_normal([256, num_classes], stddev=0.1), name="outy_w")
			output_weights = tf.get_variable(name="outy_w", shape=[256, num_classes], dtype=tf.float32, initializer=None)
			output_biases = tf.Variable(tf.zeros([num_classes]), name="outp_b")


				# reshape input data 
				X = tf.reshape(data, shape=[-1, image_size, image_size, num_channels])

				# convolution layer                                 kernel stride = 1   padded with zeros
				conv_1 = tf.tanh(tf.nn.conv2d(X, conv_1_weights, strides=[1, 1, 1, 1], padding='SAME') + conv_1_biases)

				# max pooling layer                  kernel size = 2     kernel stride = 2   padded with zeros
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

				# reshape convolution layer to feed into fully connected
				layer_shape = norm_3.get_shape()
				# extract number of features from shape
				num_features = layer_shape[1:4].num_elements()
				# reshape layer using number of features
				reshaped = tf.reshape(norm_3, [-1, num_features])

				# fully connected layer
				fully_conn_1 = tf.tanh(tf.matmul(reshaped, fully_conn_1_weights) + fully_conn_1_biases) 

				# dropout layer
				fully_conn_1 = tf.nn.dropout(fully_conn_1, dropout)

				# Output layer
				output = tf.tanh(tf.matmul(fully_conn_1, output_weights) + output_biases)

				return output


def CNN_Architecture_C():

			# define network weights and biases
			#					[kernel size, kernel size, num input channels, num output channels]
			#conv_1_weights = tf.Variable(tf.truncated_normal([3, 3, num_channels, 64], stddev=0.1), name="conv1y_w")
			conv_1_weights = tf.get_variable(name="conv1y_w", shape=[3, 3, num_channels, 64], dtype=tf.float32, initializer=None)
			conv_1_biases = tf.Variable(tf.zeros([64]), name="conv1y_b")

			#conv_2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), name="conv2y_w")
			conv_2_weights = tf.get_variable(name="conv2y_w", shape=[3, 3, 64, 128], dtype=tf.float32, initializer=None)
			conv_2_biases = tf.Variable(tf.zeros([128]), name="conv2y_b")

			# 8 * 8 because of 3 max pooling layers with stride = 2 : 64 -> 32 -> 16 -> 8
			#fully_conn_1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 256, 512], stddev=0.1),name="dense1y_w") 
			fully_conn_1_weights = tf.get_variable(name="dense1p_w", shape=[16 * 16 * 128, 256], dtype=tf.float32, initializer=None)
			fully_conn_1_biases = tf.Variable(tf.zeros([256]), name="dense1y_b")

			#fully_conn_2_weights = tf.Variable(tf.truncated_normal([512, 512], stddev=0.1),name="dense2y_w") 
			fully_conn_2_weights = tf.get_variable(name="dense2y_w", shape=[256, 256], dtype=tf.float32, initializer=None)
			fully_conn_2_biases = tf.Variable(tf.zeros([256]), name="dense2y_b")

			#output_weights = tf.Variable(tf.truncated_normal([512, num_classes], stddev=0.1), name="outy_w")
			output_weights = tf.get_variable(name="outy_w", shape=[256, num_classes], dtype=tf.float32, initializer=None)
			output_biases = tf.Variable(tf.zeros([num_classes]), name="outp_b")



				# reshape input data 
				X = tf.reshape(data, shape=[-1, image_size, image_size, num_channels])

				# convolution layer                                 kernel stride = 1   padded with zeros
				conv_1 = tf.tanh(tf.nn.conv2d(X, conv_1_weights, strides=[1, 1, 1, 1], padding='SAME') + conv_1_biases)

				# max pooling layer                 kernel size = 2     kernel stride = 2   padded with zeros
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

				# reshape convolution layer to feed into fully connected
				layer_shape = norm_2.get_shape()
				# extract number of features from shape
				num_features = layer_shape[1:4].num_elements()
				# reshape layer using number of features
				reshaped = tf.reshape(norm_2, [-1, num_features])

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

def CNN_Architecture_D():

			#					[kernel size, kernel size, num input channels, num output channels]
			#conv_1_weights = tf.Variable(tf.truncated_normal([3, 3, num_channels, 64], stddev=0.1), name="conv1y_w")
			conv_1_weights = tf.get_variable(name="conv1y_w", shape=[3, 3, num_channels, 64], dtype=tf.float32, initializer=None)
			conv_1_biases = tf.Variable(tf.zeros([64]), name="conv1y_b")

			#conv_2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), name="conv2y_w")
			conv_2_weights = tf.get_variable(name="conv2y_w", shape=[3, 3, 64, 128], dtype=tf.float32, initializer=None)
			conv_2_biases = tf.Variable(tf.zeros([128]), name="conv2y_b")

			#conv_3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1), name="conv3y_w")
			conv_3_weights = tf.get_variable(name="conv3y_w", shape=[3, 3, 128, 256], dtype=tf.float32, initializer=None)
			conv_3_biases = tf.Variable(tf.zeros([256]), name="conv3y_b")

			# 8 * 8 because of 3 max pooling layers with stride = 2 : 64 -> 32 -> 16 -> 8
			#fully_conn_1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 256, 512], stddev=0.1),name="dense1y_w") 
			fully_conn_1_weights = tf.get_variable(name="dense1p_w", shape=[8 * 8 * 256, 512], dtype=tf.float32, initializer=None)
			fully_conn_1_biases = tf.Variable(tf.zeros([512]), name="dense1y_b")

			#fully_conn_2_weights = tf.Variable(tf.truncated_normal([512, 512], stddev=0.1),name="dense2y_w") 
			fully_conn_2_weights = tf.get_variable(name="dense2y_w", shape=[512, 512], dtype=tf.float32, initializer=None)
			fully_conn_2_biases = tf.Variable(tf.zeros([512]), name="dense2y_b")

			#output_weights = tf.Variable(tf.truncated_normal([512, num_classes], stddev=0.1), name="outy_w")
			output_weights = tf.get_variable(name="outy_w", shape=[512, num_classes], dtype=tf.float32, initializer=None)
			output_biases = tf.Variable(tf.zeros([num_classes]), name="outp_b")


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

				# reshape convolution layer to feed into fully connected
				layer_shape = norm_3.get_shape()
				# extract number of features from shape
				num_features = layer_shape[1:4].num_elements()
				# reshape layer using number of features
				reshaped = tf.reshape(norm_3, [-1, num_features])

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

def My_Additional_CNN():

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
			conv_4_biases = tf.Variable(tf.zeros([256]), name="conv4y_b")

			#fully_conn_1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 256, 512], stddev=0.1),name="dense1y_w") 
			fully_conn_1_weights = tf.get_variable(name="dense1p_w", shape=[8 * 8 * 512, 1024], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			fully_conn_1_biases = tf.Variable(tf.zeros([512]), name="dense1y_b")

			#fully_conn_2_weights = tf.Variable(tf.truncated_normal([512, 512], stddev=0.1),name="dense2y_w") 
			fully_conn_2_weights = tf.get_variable(name="dense2y_w", shape=[1024, 512], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			fully_conn_2_biases = tf.Variable(tf.zeros([512]), name="dense2y_b")

			#output_weights = tf.Variable(tf.truncated_normal([512, num_classes], stddev=0.1), name="outy_w")
			output_weights = tf.get_variable(name="outy_w", shape=[512, num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
			output_biases = tf.Variable(tf.zeros([num_classes]), name="outp_b")


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