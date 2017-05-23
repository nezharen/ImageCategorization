import os
import tensorflow as tf

def read_image_label_list_file_and_convert_image_to_tfrecord(directory, filename):
    global sess
    global converted_to_tfrecord

    f = open(os.path.join(directory, filename), 'r')
    image_filenames = []
    image_labels = []
    for line in f:
        image_id, image_label = line[:-2].split(' ')
        if not converted_to_tfrecord:
            image_content = tf.read_file(os.path.join(directory, image_id + '.jpg'))
            image = tf.image.decode_jpeg(image_content)
            resized_image = tf.cast(tf.image.resize_images(image, [250, 250]), tf.uint8)
            image_bytes = sess.run(resized_image).tobytes()
            example = tf.train.Example(features = tf.train.Features(feature = {
                'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [int(image_label)])),
                'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_bytes]))
            }))
            writer = tf.python_io.TFRecordWriter(os.path.join(directory, image_id + '.tfrecord'))
            writer.write(example.SerializeToString())
            writer.close()
            print('generated ' + image_id + '.tfrecord')
        image_filenames.append(os.path.join(directory, image_id + '.tfrecord'))
    return image_filenames


#restore from checkpoint or init variables
sess = tf.Session()
converted_to_tfrecord = False
ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
if ckpt and ckpt.model_checkpoint_path:
    converted_to_tfrecord = True

#convert image to tfrecord and read
images_filename = read_image_label_list_file_and_convert_image_to_tfrecord(os.path.join('ic-data', 'train'), 'train.label')
filename_queue = tf.train.string_input_producer(images_filename)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    })
image_raw = tf.decode_raw(features['image'], tf.uint8)
image = tf.image.rgb_to_grayscale(tf.reshape(image_raw, [250, 250, 3]))
image = tf.cast(image, tf.float32) * (1. / 255)
label = tf.cast(features['label'], tf.int32)
min_after_dequeue = 500
batch_size = 3
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [image, label], batch_size = batch_size, capacity = capacity,
    min_after_dequeue = min_after_dequeue)
conv2d_layer_one = tf.contrib.layers.convolution2d(
    image_batch,
    num_outputs = 32,
    kernel_size = (5, 5),
    activation_fn = tf.nn.relu,
    stride = (2, 2),
    trainable = True)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, 'train.ckpt')
print sess.run(conv2d_layer_one)

coord.request_stop()
coord.join(threads)
sess.close()
