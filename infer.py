import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image as kerasimage
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

def read_image_list_file(directory, filename):
    model = ResNet50(weights='imagenet')

    f = open(os.path.join(directory, filename), 'rU')
    image_filenames = []
    for line in f:
        image_id = line.rstrip('\n')
        resized_image = kerasimage.load_img(os.path.join(directory, image_id + '.jpg'), target_size=(224, 224))
        resized_image = kerasimage.img_to_array(resized_image)
        resized_image = np.expand_dims(resized_image, axis=0)
        resized_image = preprocess_input(resized_image)
        preds = model.predict(resized_image)
        image_bytes = np.reshape(preds, [1000])
        example = tf.train.Example(features = tf.train.Features(feature = {
            'image': tf.train.Feature(float_list = tf.train.FloatList(value = image_bytes))
        }))
        writer = tf.python_io.TFRecordWriter(os.path.join(directory, image_id + '.tfrecord'))
        writer.write(example.SerializeToString())
        writer.close()
        print('generated ' + image_id + '.tfrecord')
        image_filenames.append(os.path.join(directory, image_id + '.tfrecord'))
    f.close()
    return image_filenames


images_filename = read_image_list_file(os.path.join('ic-data', 'check'), 'check.doc.list')
tf.reset_default_graph()

filename_queue = tf.train.string_input_producer(
    images_filename,
    shuffle = False)
reader = tf.TFRecordReader()
key, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features = {
        'image': tf.FixedLenFeature([1000], tf.float32),
    })
image = tf.reshape(features['image'], [1000])
min_after_dequeue = 500
batch_size = 3
capacity = min_after_dequeue + 3 * batch_size
image_batch, filename_batch = tf.train.batch(
    [image, key], batch_size = batch_size,
    capacity = capacity)

hidden_layer_four = tf.contrib.layers.fully_connected(
    inputs = image_batch,
    num_outputs = 4096,
    trainable = False)
final_layer = tf.contrib.layers.fully_connected(
    inputs = hidden_layer_four,
    num_outputs = 12,
    trainable = False)


sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord)
saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('restored from ' + ckpt.model_checkpoint_path)
    labeled_filenames = {}
    over = False
    result_file = open('result.list', 'w')
    while not over:
        filenames, labels = sess.run((filename_batch, tf.argmax(input = final_layer, axis = 1) + 1))
        for i in range(0, batch_size):
            if filenames[i] in labeled_filenames:
                result_file.close()
                over = True
                break
            labeled_filenames[filenames[i]] = labels[i]
            file_id = filenames[i].split(os.sep)[-1].split('.')[0]
            result_file.write(file_id + (' %d\n' % labels[i]))
else:
    print('Error: Can not find checkpoint. Please run train.py first.')

coord.request_stop()
coord.join(threads)
sess.close()
