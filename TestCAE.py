import os
import utility_functions as utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def load_model(model_file_name):
    # Load model
    graph = utils.load_graph(model_file_name)
    input = graph.get_tensor_by_name('model/input:0')
    output = graph.get_tensor_by_name('model/output/Relu:0')

    return graph, input, output

# Read test data
batch_size = 1
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
batch_x_test, _ = mnist.test.next_batch(batch_size)

# Load model
curr_dir = os.path.dirname(os.path.realpath(__file__))
model_name = 'mnist_denoising.pb'
model_path = os.path.join(curr_dir, 'logs', 'model', model_name)

graph, input_tensor, output_tensor = load_model(model_path)
sess = tf.Session(graph=graph)

noisy_data = batch_x_test + np.random.normal(0, 0.1, 784)
output_evaluated = sess.run(output_tensor, feed_dict={input_tensor:np.reshape(noisy_data, (1, 28, 28, 1))})

plt.subplot(211)
plt.imshow(np.reshape(noisy_data, (28, 28)) , cmap='gray')
plt.title('Original Data')
plt.subplot(212)
plt.imshow(np.reshape(output_evaluated, (28, 28)), cmap='gray')
plt.title('Denoised Data')
plt.show()
