from network import *
from build_vgg19 import build_vgg19
import collections

def classfiy_model(input,label,flags):
    Network = collections.namedtuple('Network', 'pred, input,label,discrim_loss, train, global_step, learning_rate,accuracy')

    with tf.variable_scope('discriminator'):
        pred = discriminator(input,flags)

    with tf.variable_scope('loss'):
        onehot_labels = tf.one_hot(label,flags.class_num)
        discrim_loss = tf.reduce_sum(tf.contrib.losses.softmax_cross_entropy(pred,onehot_labels))

    with tf.variable_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(pred, 1), label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step, flags.decay_step, flags.decay_rate,
                                                   staircase=flags.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('dicriminator_train'):

        discrim_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=flags.beta)
        discrim_train = discrim_optimizer.minimize(discrim_loss)

    return Network(
        pred=pred,
        input = input,
        label = label,
        discrim_loss = discrim_loss,
        accuracy = accuracy,
        global_step = global_step,
        train = tf.group(discrim_train, incr_global_step),
        learning_rate = learning_rate
    )

def combined_model(input1,input2,input3,label,flags):
    Network = collections.namedtuple('Network', 'per_loss, discrim_loss, train, global_step, learning_rate,y_pred, accuracy,total_loss')

    with tf.variable_scope('generator'):
        gen_input = tf.concat([input1,input2],3)
        gen_img = generator(gen_input)

    with tf.variable_scope('perceptual_loss'):
        extract_fea_gen = build_vgg19(gen_img,flags.vgg19_path,flags.vgg19_layer)
        extract_fea_target = build_vgg19(input3,flags.vgg19_path,flags.vgg19_layer)

        per_loss = tf.reduce_mean(tf.square(tf.subtract(extract_fea_gen,extract_fea_target)))

    with tf.variable_scope('discriminator'):
        dis_input = tf.concat([gen_img,input3],0)
        pred = discriminator(dis_input,flags)

    with tf.variable_scope('disc_loss'):
        dis_label = tf.concat([label, label], 0)
        onehot_labels = tf.one_hot(dis_label, flags.class_num)
        discrim_loss = tf.reduce_sum(tf.contrib.losses.softmax_cross_entropy(pred, onehot_labels))

    with tf.variable_scope('accuracy'):
        y_pred = tf.arg_max(pred,1)
        correct_prediction = tf.equal(y_pred, dis_label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.variable_scope('total_oss'):
        total_loss = flags.alpha*per_loss+flags.gamma*discrim_loss

    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step, flags.decay_step, flags.decay_rate,
                                                   staircase=flags.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('train'):

        discrim_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=flags.beta)
        train_op = discrim_optimizer.minimize(total_loss)

    return Network(
        y_pred= y_pred,
        per_loss = per_loss,
        discrim_loss = discrim_loss,
        total_loss = total_loss,
        accuracy = accuracy,
        global_step = global_step,
        train = tf.group(train_op, incr_global_step),
        learning_rate = learning_rate
    )






