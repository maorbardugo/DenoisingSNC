
import os
import re
import utility_functions as utils
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#   ---------------------------------
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 100
num_record = 20
n_epochs = 50
curr_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint_path = os.path.join(curr_dir, 'logs', 'model')
model_name = 'mnist_denoising.pb'
model_inputs = ['model/input']
model_output = 'model/output/Relu'

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
# y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

# This is
logs_path = "./logs/"


#   ---------------------------------
def activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
    x: Tensor
    Returns:
    nothing
    """
    tensor_name = re.sub('%s', '', x.op.name)
    tf.summary.histogram(tensor_name + '/hist', x)


def filter_summary(x, num_filters, order=[3, 1, 2, 0]):
    """Helper to create summaries for activations.

    Creates a summary that provides an image of activations.

    Args:
    x: Tensor
    Returns:
    nothing
    """
    tensor_name = re.sub('%s', '', x.op.name)
    kernel_transposed = tf.transpose(x, order)
    tf.summary.image(tensor_name, kernel_transposed, max_outputs=num_filters)


"""
We start by creating the layers with name scopes so that the graph in
the tensorboard looks meaningful
"""


#   ---------------------------------
def conv2d(input, name, kshape, strides=[1, 1, 1, 1]):
    with tf.name_scope(name):
        W = tf.get_variable(name='w_' + name,
                            shape=kshape,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_' + name,
                            shape=[kshape[3]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        out = tf.nn.conv2d(input, W, strides=strides, padding='SAME')
        out = tf.nn.bias_add(out, b)
        out = tf.nn.relu(out)
        filter_summary(W, kshape[3], order=[3, 1, 0, 2])

        return out


# ---------------------------------
def deconv2d(input, name, kshape, n_outputs, strides=[1, 1]):
    with tf.name_scope(name):
        out = tf.contrib.layers.conv2d_transpose(input,
                                                 num_outputs=n_outputs,
                                                 kernel_size=kshape,
                                                 stride=strides,
                                                 padding='SAME',
                                                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(
                                                     uniform=False),
                                                 biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                 activation_fn=tf.nn.relu)
        return out


#   ---------------------------------
def maxpool2d(x, name, kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.name_scope(name):
        out = tf.nn.max_pool(x,
                             ksize=kshape,  # size of window
                             strides=strides,
                             padding='SAME')
        return out


#   ---------------------------------
def upsample(input, name, factor=[2, 2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
        return out


#   ---------------------------------
def fullyConnected(input, name, output_size):
    with tf.name_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        W = tf.get_variable(name='w_' + name,
                            shape=[input_size, output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_' + name,
                            shape=[output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        input = tf.reshape(input, [-1, input_size])
        out = tf.nn.relu(tf.add(tf.matmul(input, W), b))
        activation_summary(W)
        activation_summary(b)
        return out


#   ---------------------------------
def dropout(input, name, keep_rate):
    with tf.name_scope(name):
        out = tf.nn.dropout(input, keep_rate)
        return out


#   ---------------------------------
# Let us now design the autoencoder
def ConvAutoEncoder(x):
    """
    We want to get dimensionality reduction of 784 to 196
    Layers:
        input --> 28, 28 (784)
        conv1 --> kernel size: (5,5), n_filters:25 ???make it small so that it runs fast
        pool1 --> 14, 14, 25
        dropout1 --> keeprate 0.8
        reshape --> 14*14*25
        FC1 --> 14*14*25, 14*14*5
        dropout2 --> keeprate 0.8
        FC2 --> 14*14*5, 196 --> output is the encoder vars
        FC3 --> 196, 14*14*5
        dropout3 --> keeprate 0.8
        FC4 --> 14*14*5,14*14*25
        dropout4 --> keeprate 0.8
        reshape --> 14, 14, 25
        deconv1 --> kernel size:(5,5,25), n_filters: 25
        upsample1 --> 28, 28, 25
        FullyConnected (outputlayer) -->  28* 28* 25, 28 * 28
        reshape --> 28*28
    """
    input = tf.reshape(x, shape=[-1, 28, 28, 1], name='input')

    # Encoding part
    c1 = conv2d(input, name='c1', kshape=[5, 5, 1, 6])  # 6 filters
    p1 = maxpool2d(c1, name='p1')
    do1 = dropout(p1, name='do1', keep_rate=0.95)
    do1 = tf.reshape(do1, shape=[-1, 14 * 14 * 6])
    fc1 = fullyConnected(do1, name='fc1', output_size=8 * 8 * 4)
    do2 = dropout(fc1, name='do2', keep_rate=0.95)
    fc2 = fullyConnected(do2, name='fc2', output_size=50)

    # Decoding part
    fc3 = fullyConnected(fc2, name='fc3', output_size=8 * 8 * 4)
    do3 = dropout(fc3, name='do3', keep_rate=0.95)
    fc4 = fullyConnected(do3, name='fc4', output_size=14 * 14 * 6)  # 8 filters
    do4 = dropout(fc4, name='do3', keep_rate=0.95)
    do4 = tf.reshape(do4, shape=[-1, 14, 14, 6])  # 8 filters
    dc1 = deconv2d(do4, name='dc1', kshape=[5, 5], n_outputs=6)  # 6 filters
    up1 = upsample(dc1, name='up1', factor=[2, 2])
    output = fullyConnected(up1, name='output', output_size=28 * 28)
    return output


#   ---------------------------------
def train_network(x):
    with tf.variable_scope('model'):
        prediction = ConvAutoEncoder(x)

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.square(tf.subtract(prediction, x)))

    with tf.variable_scope('opt'):
        optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Create a summary to monitor cost tensor
    tf.summary.scalar("cost", loss)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Define session
    sess = tf.Session()

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Init session
    sess.run(tf.global_variables_initializer())

    # create log writer object
    writer_train = tf.summary.FileWriter(logs_path + 'train/', graph=tf.get_default_graph())
    writer_validation = tf.summary.FileWriter(logs_path + 'validation', graph=tf.get_default_graph())

    for epoch in range(n_epochs):
        avg_cost = 0
        avg_cost_test = 0
        n_batches = int(mnist.train.num_examples / batch_size)

        # Loop over all batches
        for i in range(n_batches):
            batch_x, _ = mnist.train.next_batch(batch_size)
            batch_x_test, _ = mnist.test.next_batch(batch_size)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, train_loss, summary = sess.run([optimizer, loss, merged_summary_op], feed_dict={x: batch_x})

            # write log
            if not (i % num_record):
                validation_loss, summary_validation = sess.run([loss, merged_summary_op], feed_dict={x: batch_x_test})
                writer_train.add_summary(summary, epoch * n_batches + i)
                writer_validation.add_summary(summary_validation, epoch * n_batches + i)

            print('Train Loss = ', str(train_loss), 'Validation Loss = ', str(validation_loss))

        print('Finished epoch #', str(epoch))

        # Display logs per epoch step
        # print('Epoch', epoch + 1, ' / ', n_epochs, 'cost:', avg_cost)
        # print('Epoch', epoch + 1, ' / ', n_epochs, 'cost:', avg_cost_test)

    print('Optimization Finished')
    global_step = n_epochs*n_batches
    saver.save(sess, checkpoint_path, global_step=global_step)

    # save model
    checkpoint = utils.save_model(sess, saver, checkpoint_path, global_step)
    utils.deploy_model(sess, model_name, model_output, model_inputs, checkpoint_path, checkpoint)


train_network(x)
