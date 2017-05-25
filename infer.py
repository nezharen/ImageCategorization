import os
import tensorflow as tf

def read_image_list_file(directory, filename):
    f = open(os.path.join(directory, filename), 'rU')
    image_filenames = []
    for line in f:
        image_id = line[:-1]
        image_filenames.append(os.path.join(directory, image_id + '.jpg'))
    f.close()
    return image_filenames


sess = tf.Session()
images_filename = read_image_list_file(os.path.join('ic-data', 'check'), 'check.doc.list')
filename_queue = tf.train.string_input_producer(
    images_filename,
    shuffle = False)
reader = tf.WholeFileReader()
key, image_raw = reader.read(filename_queue)
image = tf.image.decode_jpeg(image_raw)
image = tf.image.resize_images(image, [250, 250])
image = tf.reshape(image, [250, 250, 3])
image = tf.cast(image, tf.float32) / 255
min_after_dequeue = 500
batch_size = 3
capacity = min_after_dequeue + 3 * batch_size
image_batch, filename_batch = tf.train.batch(
    [image, key], batch_size = batch_size,
    capacity = capacity)
conv2d_layer_one = tf.contrib.layers.conv2d(
    inputs = image_batch,
    num_outputs = 32,
    kernel_size = (5, 5),
    stride = (2, 2),
    padding = 'SAME',
    trainable = False)
pool_layer_one = tf.contrib.layers.max_pool2d(
    inputs = conv2d_layer_one,
    kernel_size = [2, 2],
    stride = [2, 2],
    padding = 'SAME')
conv2d_layer_two = tf.contrib.layers.conv2d(
    inputs = pool_layer_one,
    num_outputs = 64,
    kernel_size = (5, 5),
    stride = (1, 1),
    padding = 'SAME',
    trainable = False)
pool_layer_two = tf.contrib.layers.max_pool2d(
    inputs = conv2d_layer_two,
    kernel_size = [2, 2],
    stride = [2, 2],
    padding = 'SAME')
flattened_layer_two = tf.reshape(
    pool_layer_two,
    [
        batch_size,
        -1
    ])
hidden_layer_three = tf.contrib.layers.fully_connected(
    inputs = flattened_layer_two,
    num_outputs = 512,
    trainable = False)
hidden_layer_three = tf.contrib.layers.dropout(
    inputs = hidden_layer_three)
final_layer = tf.contrib.layers.fully_connected(
    inputs = hidden_layer_three,
    num_outputs = 12,
    trainable = False)

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
