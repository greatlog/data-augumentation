import tensorflow as tf
import numpy as np
import random

def dis_data_loader(flags,data_list):

    file_list = tf.convert_to_tensor(data_list['data'],tf.string)
    label_list = tf.convert_to_tensor(data_list['label'],tf.int64)

    if flags.mode == 'train':
        queue = tf.train.slice_input_producer([file_list,label_list])
    elif flags.mode =='test':
        queue = tf.train.slice_input_producer([file_list, label_list],num_epochs=1)

    img_raw = tf.read_file(queue[0])

    img = tf.image.decode_jpeg(img_raw,channels=3)
    img = tf.cast(img,tf.float32)
    label = queue[1]

    img_batch, label_batch = tf.train.shuffle_batch([img,label],batch_size=flags.batch_size,
                                                    shapes=[[flags.input_size,flags.input_size,3],[]],
                                                    num_threads=8,capacity=flags.batch_size*10,
                                                    min_after_dequeue=flags.batch_size)
    print('trian data is loaded')

    return img_batch, label_batch



def gan_data_loader(data_list,flags):
    list_label = np.array(data_list['label']).astype(np.int64)

    list1 = data_list['data']
    list2,list3= [],[]
    for i in range(len(list1)):
        single_label = list_label[i]
        index = np.where(list_label==single_label)[0]
        ind = random.sample(index,2)
        list2.append(list1[ind[0]])
        list3.append(list1[ind[1]])

    list1 = tf.convert_to_tensor(list1,tf.string)
    list2 = tf.convert_to_tensor(list2,tf.string)
    list3 = tf.convert_to_tensor(list3,tf.string)
    list_label = tf.convert_to_tensor(list_label,tf.int64)

    if flags.mode == 'train':
        filename = tf.train.slice_input_producer([list1, list2, list3, list_label])
    if flags.mode == 'test':
        filename = tf.train.slice_input_producer([list1, list2, list3, list_label],num_epochs=1)

    img1 = tf.read_file(filename[0])
    img2 = tf.read_file(filename[1])
    img3 = tf.read_file(filename[2])

    img1 = tf.reshape(tf.image.decode_jpeg(img1), [flags.input_size, flags.input_size, 3])
    img2 = tf.reshape(tf.image.decode_jpeg(img2), [flags.input_size, flags.input_size, 3])
    img3 = tf.reshape(tf.image.decode_jpeg(img3), [flags.input_size, flags.input_size, 3])

    label = filename[3]

    img1_batch,img2_batch,img3_batch,label_batch = tf.train.shuffle_batch([img1,img2,img3,label],batch_size=flags.batch_size,
                                                                          shapes=[[flags.input_size,flags.input_size,3],
                                                                                  [flags.input_size, flags.input_size,3],
                                                                                  [flags.input_size, flags.input_size,3],
                                                                                  []],
                                                                          num_threads=8,capacity=flags.batch_size*10,
                                                                          min_after_dequeue=flags.batch_size)



    return img1_batch, img2_batch,img3_batch,label_batch





