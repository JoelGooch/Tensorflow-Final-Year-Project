import tensorflow as tf
import numpy as np
import os
import pickle
import datetime
import sys

def calc_accuracy(predictions, labels, verbose=True):
    '''This function return the accuracy for the current batch.

    Because the network has only one neuron as output, and the output
    is considered continous, the accuracy is measured through RMSE
    @param predictions the output of the network for each image passed
    @param labels the correct category (target) for the immage passed
    @param verbose if True prints information on terminal
    @return it returns the RMSE (Root Mean Squared Error)
    '''

    # Convert back to degree
    predictions_degree = predictions * 180
    predictions_degree -= 90
    labels_degree = labels * 180
    labels_degree -= 90
    # RMSE = Root Mean Squared Error
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


    # It returns the RMSE
    return np.sqrt(np.sum(np.square(predictions_degree - labels_degree), dtype=np.float32) * 1 / predictions.shape[0])


def main():

    total_epochs = 200
    batch_size = 64
    learning_rate = 0.05


    image_size = 64 # images are 64x64x3
    num_channels = 3 # RGB
    num_classes = 1 # regression
    pickle_directory = "C:/Users/Joel Gooch/Desktop/Final Year/PRCO304/data//prima_pitch_p1_out.pickle"
    
    with open(pickle_directory, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
        training_data = data[b'training_dataset']
        training_labels = data[b'training_label']
        testing_data = data[b'validation_dataset']
        testing_labels = data[b'validation_label']
    

    training_data = training_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    training_labels = training_labels.reshape((-1, 1)).astype(np.float32)

    testing_data = testing_data.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    testing_labels = testing_labels.reshape((-1, 1)).astype(np.float32)

    # normalize data
    training_data -= 127 
    testing_data -= 127

    graph = tf.Graph()
    with graph.as_default():

        '''
        # select and print random image from testing set 
        random = np.random.randint(0, testing_data.shape[0])
        img = np.copy(testing_data[random, :] + 127)
        test_class = testing_labels[random, :]
        img = img.astype(np.uint8)
        img = img.reshape(64, 64, num_channels)
        cv2.imshow('image', img)
        print("image class = " + str(test_class))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        print("Initialising Tensorflow Variables...")

        # define placeholder variables
        x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
        y = tf.placeholder(tf.float32, shape=(None, num_classes))

        conv1_weights = tf.Variable(tf.truncated_normal([3, 3, num_channels, 64], stddev=0.1), name="conv1y_w")
        conv1_biases = tf.Variable(tf.zeros([64]), name="conv1y_b")

        conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), name="conv2y_w")
        conv2_biases = tf.Variable(tf.random_normal(shape=[128]), name="conv2y_b")

        dense1_weights = tf.Variable(tf.truncated_normal([16 * 16 * 128, 128], stddev=0.1),name="dense1y_w")  # was [5*5*256, 1024]
        dense1_biases = tf.Variable(tf.random_normal(shape=[128]), name="dense1y_b")

        # Output layer
        layer_out_weights = tf.Variable(tf.truncated_normal([128, num_classes], stddev=0.1), name="outy_w")
        layer_out_biases = tf.Variable(tf.random_normal(shape=[num_classes]), name="outy_b")

        # dropout (keep probability)
        keep_prob = tf.placeholder(tf.float32)


        def CNN_Model(data, _dropout=1.0):

            X = tf.reshape(data, shape=[-1, image_size, image_size, num_channels])

            conv1 = tf.sigmoid(tf.nn.conv2d(X, conv1_weights, strides=[1, 1, 1, 1], padding='SAME') + conv1_biases)

            # Max Pooling (down-sampling)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # Apply Normalization
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            #norm1 = tf.nn.dropout(norm1, _dropout)

            # Convolution Layer 2
            conv2 = tf.sigmoid(tf.nn.conv2d(norm1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME') + conv2_biases)

            # Max Pooling (down-sampling)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Apply Normalization
            norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
            # Apply Dropout
            norm2 = tf.nn.dropout(norm2, _dropout)

            # Fully connected layer 1
            dense1 = tf.reshape(norm2, [-1, dense1_weights.get_shape().as_list()[0]])  # Reshape conv3
            dense1 = tf.sigmoid(tf.matmul(dense1, dense1_weights) + dense1_biases) 
            dense1 = tf.nn.dropout(dense1, _dropout)

            # Output layer
            output = tf.sigmoid(tf.matmul(dense1, layer_out_weights) + layer_out_biases)

            return output


        model_output = CNN_Model(x, keep_prob)

        print(model_output.get_shape)
        
        #accuracy = calc_accuracy(model_output, y)

        loss = tf.nn.l2_loss(model_output - y)
        

        #Adding the regularization terms to the loss
        beta = 5e-4
        loss += (beta * tf.nn.l2_loss(conv1_weights)) 
        loss += (beta * tf.nn.l2_loss(conv2_weights)) 
        #loss += (beta * tf.nn.l2_loss(conv3_weights)) 
        #loss += (beta * tf.nn.l2_loss(conv4_weights))
        loss += (beta * tf.nn.l2_loss(dense1_weights))
        #loss += (beta * tf.nn.l2_loss(dense2_weights))
        loss += (beta * tf.nn.l2_loss(layer_out_weights))
        
        
        loss_summary = tf.summary.scalar("loss", loss)

        global_step = tf.Variable(0, trainable=False)

        #learning_rate = tf.train.exponential_decay(0.0125, global_step, 15000, 0.1, staircase=True)
        #lrate_summary = tf.summary.scalar("learning rate", learning_rate)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        #optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
        #optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
        #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95).minimize(loss, global_step=global_step)


        saver = tf.train.Saver()

        with tf.Session(graph=graph) as session:
            merged_summaries = tf.summary.merge_all()
            now = datetime.datetime.now()
            log_path = "/tmp/Prima/log/" + str(now.hour) + str(now.minute) + str(now.second)
            writer_summaries = tf.summary.FileWriter(log_path, graph)

            save_path = '/tmp/Prima/checkpoints/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            try:
                print("Trying to restore last checkpoint ...")
                # Use TensorFlow to find the latest checkpoint - if any.
                last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_path)

                # Try and load the data in the checkpoint.
                saver.restore(session, save_path=last_chk_path)
                print("Restored checkpoint from:", last_chk_path)
            except:
                print("Failed to restore checkpoint. Initializing variables instead.")
                session.run(tf.global_variables_initializer())


            for epoch in range(total_epochs):

                offset = (epoch * batch_size) % (training_labels.shape[0] - batch_size)
                batch_data = training_data[offset:(offset + batch_size), :, :, :]
                batch_labels = training_labels[offset:(offset + batch_size)]

                #print("batch labels = " + str(batch_labels))

                feed_dict = {x: batch_data, y: batch_labels, keep_prob: 0.5}
                _, l, predictions, my_summary = session.run([optimizer, loss, model_output, merged_summaries], 
                                                feed_dict=feed_dict)
                writer_summaries.add_summary(my_summary, epoch)

                #print("output = " + str(pred_class))

                if (epoch % 100 == 0):
                    print("")
                    print("Loss at epoch: ", epoch, "of ", str(total_epochs) ," is " , l)
                    print("Global Step: " + str(global_step.eval()))
                    acc = calc_accuracy(predictions, batch_labels)
                    print("Accuracy: %.2f" % acc)
                    #print("Learning Rate: " + str(learning_rate.eval()))
                    print("Minibatch size: " + str(batch_labels.shape))

                #if (epoch % 10000 == 0):
                #    saver.save(session, save_path=save_path, global_step=global_step)
                #    print("Saved Checkpoint")

            feed_dict = {x: testing_data, y: testing_labels, keep_prob: 1.0}
            _, l, predictions, my_summary = session.run([optimizer, loss, model_output, merged_summaries], 
                                    feed_dict=feed_dict)

            print("# Test RMSE: %.2f" % calc_accuracy(predictions, testing_labels, True))
            saver.save(session, save_path=save_path, global_step=global_step)
            print("Saved Checkpoint")

            
if __name__ == "__main__":
    main()