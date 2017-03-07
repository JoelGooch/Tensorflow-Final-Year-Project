import tensorflow as tf
import numpy as np
import os
import pickle
import datetime
import cv2


def train_network(self, num_epochs, batch_size, learning_rate, learning_algo, save_dir, save_interval):
        image_size = 32 # images are 32x32x3
        num_channels = 3 # RGB
        num_classes = 10 # 10 possible classes

        num_training_files = 5 # CIFAR10 training data is split into 5 files
        num_images_per_file = 10000
        num_training_images_total = num_training_files * num_images_per_file
        num_testing_images_total = 10000

        pickle_directory = "C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/data/CIFAR-10/cifar-10-batches-py/"

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

        # normalize data
        training_set -= 127 
        testing_set -= 127

        graph = tf.Graph()
        with graph.as_default():

            self.logMessage.emit('Initialising Tensorflow Variables... \n')

            # define placeholder variables
            x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
            y = tf.placeholder(tf.float32, shape=(None, num_classes))

            # stores the class integer values 
            labels_class = tf.argmax(y, dimension=1)


            # define network structure
            #[conv_kernel_size, conv_kernel_size, num_channels, num_output_filters]
            conv1_weights = tf.Variable(tf.truncated_normal([5, 5, num_channels, 64], stddev=0.05), name="conv1_weights")
            conv1_biases = tf.Variable(tf.zeros([64]), name="conv1_biases")

            conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=0.05), name="conv2_weights")
            conv2_biases = tf.Variable(tf.random_normal(shape=[64]), name='conv2_biases')

            dense1_weights = tf.Variable(tf.truncated_normal([8 * 8 * 64, 256], stddev=0.05), name="dense1_weights")
            dense1_biases = tf.Variable(tf.random_normal(shape=[256]), name="dense1_biases")

            dense2_weights = tf.Variable(tf.truncated_normal([256, 128], stddev=0.050), name='dense2_weights')
            dense2_biases = tf.Variable(tf.random_normal(shape=[128]), name='dense2_biases')

            output_weights = tf.Variable(tf.truncated_normal([128, num_classes], stddev=0.05), name="output_weights")
            output_biases = tf.Variable(tf.random_normal(shape=[num_classes]), name="output_biases")

            keep_prob = tf.placeholder(tf.float32)


            def CIFAR10_CNN_Model(data, _dropout=1.0):

                X = tf.reshape(data, shape=[-1, image_size, image_size, num_channels])
                self.logMessage.emit('Shape of Input Data: {}'.format(str(X.get_shape())))

                # conv layer 1 
                conv1 = tf.nn.relu(tf.nn.conv2d(X, conv1_weights, strides=[1, 1, 1, 1], padding='SAME') + conv1_biases)
                self.logMessage.emit('Shape of Conv1: {}'.format(str(conv1.get_shape())))

                # max pooling 1 
                pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

                # apply normalization and dropout
                norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
                #norm1 = tf.nn.dropout(norm1, _dropout)

                # conv layer 2
                conv2 = tf.nn.relu(tf.nn.conv2d(norm1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME') + conv2_biases)

                # max pooling 2
                pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

                # apply normalization and dropout
                norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
                #norm2 = tf.nn.dropout(norm2, _dropout)

                # fully connected layer 1
                dense1 = tf.reshape(norm2, [-1, dense1_weights.get_shape().as_list()[0]])
                dense1 = tf.nn.relu(tf.matmul(dense1, dense1_weights) + dense1_biases)
                #dense1 = tf.nn.dropout(dense1, _dropout)

                dense2 = tf.nn.relu(tf.matmul(dense1, dense2_weights) + dense2_biases)

                output = tf.matmul(dense2, output_weights) + output_biases

                return output


            model_output = CIFAR10_CNN_Model(x, keep_prob)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model_output, labels=y)

            loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

        
            #Adding the regularization terms to the loss
            #beta = 5e-4
            #loss += (beta * tf.nn.l2_loss(conv1_weights)) 
            #loss += (beta * tf.nn.l2_loss(conv2_weights)) 
            #loss += (beta * tf.nn.l2_loss(conv3_weights)) 
            #loss += (beta * tf.nn.l2_loss(conv4_weights))
            #loss += (beta * tf.nn.l2_loss(dense1_weights))
            #loss += (beta * tf.nn.l2_loss(dense2_weights))
            #loss += (beta * tf.nn.l2_loss(output_weights))
        

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

            network_pred_class = tf.argmax(model_output, dimension=1)
            correct_prediction = tf.equal(network_pred_class, labels_class)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

            saver = tf.train.Saver()


        with tf.Session(graph=graph) as session:
            merged_summaries = tf.summary.merge_all()
            now = datetime.datetime.now()
            log_path = "C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/tmp/CIFAR10/log/" + str(now.hour) + str(now.minute) + str(now.second)
            save_path = os.path.join(save_dir, 'CIFAR10')
            writer_summaries = tf.summary.FileWriter(log_path, graph)

            try:

                txtOutputLog.emit('Trying to restore last checkpoint ...')
                # Use TensorFlow to find the latest checkpoint - if any.
                last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)

                # Try and load the data in the checkpoint.
                saver.restore(sess=session, save_path=save_path)
                self.logMessage.emit('Restored checkpoint from: {} '.format(last_chk_path))
            except:
                self.logMessage.emit('Failed to restore checkpoint. Initializing variables instead.')
                session.run(tf.global_variables_initializer())


            for epoch in range(num_epochs):
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

                epochProg = (epoch / num_epochs) * 100
                self.epochProgress.emit(epochProg)

                if save_interval != 0:
                    if (epoch % save_interval == 0):
                        saver.save(sess=session, save_path=save_path, global_step=global_step)
                        print("Saved Checkpoint")

            self.epochProgress.emit(100)

            test_acc = session.run(accuracy, feed_dict={x: testing_set, y:testing_labels, keep_prob: 1.0})
            self.testSetAccuracy.emit(test_acc)
            saver.save(sess=session, save_path=save_path, global_step=global_step)
            self.logMessage.emit('\nTraining Complete')
            self.logMessage.emit('Saved Checkpoint \n')
            self.workComplete.emit()
