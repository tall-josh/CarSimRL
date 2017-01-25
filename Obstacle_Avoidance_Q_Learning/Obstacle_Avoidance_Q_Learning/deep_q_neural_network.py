# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import constants as CONST

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
    
def new_conv_layer(prev_layer, 
                   num_input_channels,
                   filter_size,
                   num_filters,
                   use_pooling=False):
    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=prev_layer,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights
    
def flatten_layer(layer):
    # Get the shape of the input  layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features
    
def new_fc_layer(prev_layer,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(prev_layer, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)
        
    return layer

        
        ##### CNN Layout ##### 
        
try: 
    tf.reset_default_graph()      
except:
    pass
        
# Conv Layer 1
filter_sz1 = 3#5
num_filters1 = 16#16

# Conv Layer 2
filter_sz2 = 5#5
num_filters2 = 36#36

# Fully connected
fc_size = 128#128

##### Data dimentions #####
image_size = CONST.STATE_MATRIX_SIZE      
image_size_flat = CONST.STATE_MATRIX_FLAT_SZ
num_channels = 1
num_classes = 5

# placeholders for input nodes
state = tf.placeholder(tf.float32, shape=[None, image_size_flat], name = 'state')

# reshape state tensor to a tensor [arbetrary_number_of_inputs, cols, rows, channels]
x_image = tf.reshape(state, [-1, image_size[0], image_size[1], num_channels])

#q_matrix = tf.placeholder(tf.float32, shape=[None, num_classes], name = 'q_matrix')
q_target = tf.placeholder(tf.float32, shape=[None, num_classes], name = 'q_target')


layer_conv1, weights_conv1 = new_conv_layer(prev_layer=x_image,
                                                 num_input_channels = num_channels,
                                                 filter_size = filter_sz1,
                                                 num_filters=num_filters1,
                                                 use_pooling=False)

layer_conv2, weights_conv2 = new_conv_layer(prev_layer=layer_conv1,
                                                 num_input_channels = num_filters1,
                                                 filter_size = filter_sz2,
                                                 num_filters=num_filters2,
                                                 use_pooling=False)

layer_flat, num_features = flatten_layer(layer_conv2)


layer_fc1 = new_fc_layer(prev_layer = layer_flat,
                         num_inputs = num_features,
                         num_outputs = fc_size,
                         use_relu=True)

q_matrix = new_fc_layer(prev_layer=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

##### CLASS PREDICTION #####

q_est_action = tf.argmax(q_matrix, dimension=1)
reduction = tf.square(q_target - q_matrix)
cost = tf.reduce_mean(reduction, reduction_indices=1)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())
#train_batch_size=64

# Counter for total number of iterations performed so far.
total_iterations = 0

def getQMat(state_flattened):
    
    state_feed = np.zeros((1,len(state_flattened)))
    state_feed[0] = state_flattened
    #q_feed = np.zeros((1,len(CONST.ACTION_AND_COSTS)))
    feed_dict = {state: state_feed}
                 
    return session.run(q_matrix, feed_dict=feed_dict)

# fit(scan.flatten(), )
def fit(state_flattened, target_qs_flattened):
    state_feed = np.zeros((1,len(state_flattened)))
    state_feed[0] = state_flattened
    q_feed = np.zeros((1,len(CONST.ACTION_AND_COSTS)))
    q_feed[0] = target_qs_flattened
    state_action_dict = {state: state_feed, q_target: q_feed}
    session.run(optimizer, feed_dict = state_action_dict)
    
def fitBatch(batch_state_flattened, batch_target_qs_flattened):
    state_feed = np.zeros((len(batch_state_flattened),len(batch_state_flattened[0])))
    q_feed = np.zeros((len(batch_state_flattened),len(CONST.ACTION_AND_COSTS)))
    for i in range(len(batch_state_flattened)):
        state_feed[i] = batch_state_flattened[i]
        q_feed[i] = batch_target_qs_flattened[i]

    state_action_dict = {state: state_feed, q_target: q_feed}  
    session.run(optimizer, feed_dict = state_action_dict)

#def fit(state_flattened, target_qs):
#    state_feed = np.zeros((1,len(state_flattened)))
#    state_feed[0] = state_flattened
#    q_feed = np.zeros((1,len(CONST.ACTION_AND_COSTS)))
#    print("state_feed: ",state_feed.shape)
#    print("q_target: ",q_target.get_shape())
#    state_action_dict = {state: state_feed, q_target: q_feed}
#    session.run(optimizer, feed_dict = state_action_dict)
    
def experienceReplay(num_iterations, experience_dict):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=experience_dict)

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
