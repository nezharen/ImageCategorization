import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image as kerasimage
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

def read_image_label_list_file_and_convert_image_to_tfrecord(directory, filename):
    global converted_to_tfrecord
    global model

    f = open(os.path.join(directory, filename), 'rU')
    image_filenames = []
    for line in f:
        image_id, image_label = line.rstrip('\n').split(' ')
        if not converted_to_tfrecord:
            resized_image = kerasimage.load_img(os.path.join(directory, image_id + '.jpg'), target_size=(224, 224))
            resized_image = kerasimage.img_to_array(resized_image)
            resized_image = np.expand_dims(resized_image, axis=0)
            resized_image = preprocess_input(resized_image)
            preds = model.predict(resized_image)
            image_bytes = np.reshape(preds, [1000])
            example = tf.train.Example(features = tf.train.Features(feature = {
                'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [int(image_label) - 1])),
                'image': tf.train.Feature(float_list = tf.train.FloatList(value = image_bytes))
            }))
            writer = tf.python_io.TFRecordWriter(os.path.join(directory, image_id + '.tfrecord'))
            writer.write(example.SerializeToString())
            writer.close()
            print('generated ' + image_id + '.tfrecord')
        image_filenames.append(os.path.join(directory, image_id + '.tfrecord'))
    f.close()
    return image_filenames


#restore from checkpoint or init variables
converted_to_tfrecord = False
ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
if ckpt and ckpt.model_checkpoint_path:
    converted_to_tfrecord = True
    model = None
else:
    model = ResNet50(weights='imagenet')

#convert image to tfrecord and read
images_filename = read_image_label_list_file_and_convert_image_to_tfrecord(os.path.join('ic-data', 'train'), 'train.label')
tf.reset_default_graph()

filename_queue = tf.train.string_input_producer(
    images_filename,
    shuffle = False
    )
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features = {
        'image': tf.FixedLenFeature([1000], tf.float32),
        'label': tf.FixedLenFeature([], tf.int64),
    })
image = tf.reshape(features['image'], [1000])
label = tf.cast(features['label'], tf.int32)
min_after_dequeue = 500
batch_size = 3
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [image, label], batch_size = batch_size, capacity = capacity,
    min_after_dequeue = min_after_dequeue)

hidden_layer_three = tf.contrib.layers.fully_connected(
    inputs = image_batch,
    num_outputs = 4096,
    trainable = True)
hidden_layer_three = tf.contrib.layers.dropout(hidden_layer_three)
hidden_layer_four = tf.contrib.layers.fully_connected(
    inputs = hidden_layer_three,
    num_outputs = 4096,
    trainable = True)
hidden_layer_four = tf.contrib.layers.dropout(hidden_layer_four)
final_layer = tf.contrib.layers.fully_connected(
    inputs = hidden_layer_four,
    num_outputs = 12,
    trainable = True)
onehot_labels = tf.one_hot(indices = label_batch, depth = 12)
loss = tf.losses.softmax_cross_entropy(
    onehot_labels = onehot_labels,
    logits = final_layer)
train_op = tf.contrib.layers.optimize_loss(
    loss = loss,
    global_step = tf.contrib.framework.get_global_step(),
    learning_rate = 0.001,
    optimizer = 'SGD')

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord)
saver = tf.train.Saver()

#sess.run(tf.global_variables_initializer())
#saver.save(sess, 'train.ckpt')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('restored from ' + ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, 'train.ckpt')

step = 0
while True:
    sess.run(train_op)
    step = step + 1
    if step % 100 == 0:
        saver.save(sess, 'train.ckpt')
        print('step = %d, loss = %f' % (step, sess.run(loss)))

coord.request_stop()
coord.join(threads)
sess.close()
