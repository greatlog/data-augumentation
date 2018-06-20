import tensorflow as tf

from tensorflow.contrib.layers import conv2d,max_pool2d,repeat

def generator(input):

    conv1 = conv2d(input, 16, 3, activation_fn=tf.nn.relu,scope='conv1')
    conv2 = conv2d(conv1, 16, 3, activation_fn=tf.nn.relu,scope='conv2')
    conv3 = conv2d(conv2, 16, 3, activation_fn=tf.nn.relu, scope='conv3')
    conv4 = conv2d(conv3, 16, 3, activation_fn=tf.nn.relu, scope='conv4')
    conv5 = conv2d(conv4, 3, 3, activation_fn=None,scope='conv5')

    return conv5

def discriminator(input,flags):
    net = repeat(input, 2, conv2d, 64, 3, scope='conv1')
    net = max_pool2d(net, [2, 2], scope='pool1')
    net = repeat(net, 2, conv2d, 128, 3, scope='conv2')
    net = max_pool2d(net, [2, 2], scope='pool2')
    net = repeat(net, 4, conv2d, 256, 3, scope='conv3')
    net = max_pool2d(net, [2, 2], scope='pool3')
    net = repeat(net, 4, conv2d, 512, 3, scope='conv4')
    net = max_pool2d(net, [2, 2], scope='pool4')
    net = repeat(net, 4, conv2d, 512, 3, scope='conv5')
    net = max_pool2d(net, [2, 2], scope='pool5')
    net = tf.layers.Flatten()(net)
    net = tf.layers.dense(net, flags.class_num)

    return net








