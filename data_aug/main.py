from model import *
from dataset import *
from ops import *
import os

Flags = tf.app.flags

# The system parameter
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
Flags.DEFINE_string('data_dir','/data/proxima_data/private_data/Xray/ccyy/x-ray/chest_jpg/','the path of data')
Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
Flags.DEFINE_boolean('pre_trained_model', False, 'If set True, the weight will be loaded but the global_step will still '
                                                 'be 0. If set False, you are going to continue the training. That is, '
                                                 'the global_step will be initiallized from the checkpoint, too')

Flags.DEFINE_boolean('is_training', True, 'Training => True, Testing => False')

Flags.DEFINE_string('vgg19_path', '/data/model_weights/luozx/data_aug/imagenet-vgg-verydeep-19.mat', 'path to checkpoint file for the vgg19')
Flags.DEFINE_string('vgg19_layer','conv5_4','layer in vgg19 to extract feature')
Flags.DEFINE_string('task', None, 'The task: discriminator, combined_model')

Flags.DEFINE_integer('batch_size', 16, 'Batch size of the input batch')
Flags.DEFINE_integer('input_size',512,'input size of the picture')
Flags.DEFINE_integer('class_num',9,'num of discriminating number')
Flags.DEFINE_integer('data_split',20000,'number of training data')

Flags.DEFINE_float('alpha',0.25,'portion of augmentation loss')
Flags.DEFINE_float('gamma',0.75,'portion of classification loss')

Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')


flags = Flags.FLAGS

if flags.output_dir is None:
    raise ValueError('The output directory is needed')

# Check the output directory to save the checkpoint
if not os.path.exists(flags.output_dir):
    os.mkdir(flags.output_dir)

# Check the summary directory to save the event
if not os.path.exists(flags.summary_dir):
    os.mkdir(flags.summary_dir)

label_kind, train_list, test_list = generate_list(flags)

if flags.mode == 'train':
    if flags.task == 'discriminator':
        image, label = dis_data_loader(flags,train_list)
        net = classfiy_model(image, label,flags)

        tf.summary.scalar('dis_loss', net.discrim_loss)
        tf.summary.scalar('accuracy', net.accuracy)
        tf.summary.scalar('learning_rate', net.learning_rate)

    elif flags.task == 'combined_model':
        data1,data2,data3,label = gan_data_loader(train_list,flags)
        net = combined_model(data1,data2,data3,label,flags)

        tf.summary.scalar('dis_loss', net.discrim_loss)
        tf.summary.scalar('per_loss',net.per_loss)
        tf.summary.scalar('total_loss', net.per_loss)
        tf.summary.scalar('learning_rate', net.learning_rate)

    else:
        raise NotImplementedError('Unknown task type')

    saver = tf.train.Saver(max_to_keep=10)

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    if flags.task == 'combined_model':
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator') + \
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    elif flags.task == 'discriminator':
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    weight_initiallizer = tf.train.Saver(var_list2)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Use superviser to coordinate all queue and summary writer
    sv = tf.train.Supervisor(logdir=flags.summary_dir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:

        if (flags.checkpoint is not None) and (flags.pre_trained_model is False):
            print('Loading model from the checkpoint...')
            checkpoint = tf.train.latest_checkpoint(flags.checkpoint)
            saver.restore(sess, checkpoint)

        elif (flags.checkpoint is not None) and (flags.pre_trained_model is True):
            print('Loading weights from the pre-trained model')
            weight_initiallizer.restore(sess, flags.checkpoint)

        print('Optimization starts!!!')
        acc_txt = os.path.join(flags.output_dir,'acc.txt')
        if os.path.exists(acc_txt):
            os.remove(acc_txt)
        txt_result = open(acc_txt,'a+')
        for step in range(flags.max_iter):
            fetches = {
                "train": net.train,
                "global_step": sv.global_step,
            }

            if ((step + 1) % flags.display_freq) == 0:
                if flags.task == 'discriminator':
                    fetches["discrim_loss"] = net.discrim_loss
                    fetches["learning_rate"] = net.learning_rate
                    fetches["global_step"] = net.global_step
                    fetches["accuracy"] = net.accuracy
                    fetches["pred"] = net.pred
                    fetches["input"] = net.input
                    fetches["label"] = net.label

                elif flags.task == 'combined_model':
                    fetches["discrim_loss"] = net.discrim_loss
                    fetches["per_loss"] = net.per_loss
                    fetches["total_loss"] = net.total_loss
                    fetches["learning_rate"] = net.learning_rate
                    fetches["global_step"] = net.global_step
                    fetches["y_pred"] = net.y_pred

            if ((step + 1) %flags.summary_freq) == 0:
                fetches["summary"] = sv.summary_op


            results = sess.run(fetches)

            if ((step + 1) % flags.summary_freq) == 0:
                print('Recording summary!!')
                sv.summary_writer.add_summary(results['summary'], results['global_step'])

            if ((step + 1) % flags.display_freq) == 0:

                if flags.task == 'discriminator':
                    print("global_step", results["global_step"])
                    print("discrim_loss", results["discrim_loss"])
                    print("learning_rate", results['learning_rate'])
                    print("accuracy",results["accuracy"])
                    txt_result.write("global_step:%d,discrim_loss:%.4f,accuracy:%.4f\n"%(results["global_step"],results["discrim_loss"],results["accuracy"]))
                elif flags.task == 'combined_model':
                    print("global_step", results["global_step"])
                    print("discrim_loss", results["discrim_loss"])
                    print("per_loss", results["per_loss"])
                    print("learning_rate", results['learning_rate'])
                    txt_result.write("global_step:%d,discrim_loss:%.4f\n,per_loss:%.4f" %
                                     (results["global_step"], results["discrim_loss"],results["per_loss"]))

            if ((step + 1) % flags.save_freq) == 0:
                print('Save the checkpoint')
                saver.save(sess, os.path.join(flags.output_dir, 'model'), global_step=sv.global_step)

        print('Optimization done!!!!!!!!!!!!')


if flags.mode == 'test':
    if flags.task == 'discriminator':
        image, label = dis_data_loader(flags, test_list)
        net = classfiy_model(image, label, flags)

        tf.summary.scalar('dis_loss', net.discrim_loss)
        tf.summary.scalar('accuracy', net.accuracy)
        tf.summary.scalar('learning_rate', net.learning_rate)

    elif flags.task == 'combined_model':
        data1, data2, data3, label = gan_data_loader(test_list, flags)
        net = combined_model(data1, data2, data3, label, flags)

        tf.summary.scalar('dis_loss', net.discrim_loss)
        tf.summary.scalar('per_loss', net.per_loss)
        tf.summary.scalar('total_loss', net.per_loss)
        tf.summary.scalar('learning_rate', net.learning_rate)

    else:
        raise NotImplementedError('Unknown task type')

    saver = tf.train.Saver(max_to_keep=10)

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    if flags.task == 'combined_model':
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator') + \
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    elif flags.task == 'discriminator':
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    weight_initiallizer = tf.train.Saver(var_list2)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Use superviser to coordinate all queue and summary writer
    sv = tf.train.Supervisor(logdir=flags.summary_dir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:

        if (flags.checkpoint is not None) and (flags.pre_trained_model is False):
            print('Loading model from the checkpoint...')
            checkpoint = tf.train.latest_checkpoint(flags.checkpoint)
            saver.restore(sess, checkpoint)

        elif (flags.checkpoint is not None) and (flags.pre_trained_model is True):
            print('Loading weights from the pre-trained model')
            weight_initiallizer.restore(sess, flags.checkpoint)

        print('Evaluation starts!!!')
        acc_txt = os.path.join(flags.output_dir, 'acc.txt')
        if os.path.exists(acc_txt):
            os.remove(acc_txt)
        txt_result = open(acc_txt, 'a+')
        total_loss = 0.0
        total_acc = 0.0
        count = 0.0
        while True:
            try:
                fetches = {
                    "global_step": sv.global_step,
                    "accuracy":net.accuracy
                }


                if flags.task == 'discriminator':
                    fetches["loss"] = net.discrim_loss

                elif flags.task == 'combined_model':
                    fetches["loss"] = net.total_loss


                results = sess.run(fetches)

                total_acc = total_acc + results["accuracy"]
                total_loss = total_loss + results["loss"]
                count = count+1

                txt_result.write("global_step:%d,loss:%.4f,accuracy:%.4f\n" % (
                results["global_step"], results["loss"], results["accuracy"]))
                print("global_step:%d,loss:%.4f,accuracy:%.4f\n" % (
                    results["global_step"], results["loss"], results["accuracy"]))

            except tf.errors.OutOfRangeError as e:
                average_loss = total_loss/count
                average_acc = total_acc/count
                txt_result.write("avergae_loss:%.4f,average_accuracy:%.4f\n" % (
                    average_loss, average_acc))
                print("avergae_loss:%.4f,average_accuracy:%.4f\n" % (
                    average_loss, average_acc))
                break

        print('Evaluation done!!!!!!!!!!!!')