import os
import tensorflow as tf

def read_image_label_list_file_and_convert_image_to_tfrecord(directory, filename):
    global sess
    global converted_to_tfrecord

    f = open(os.path.join(directory, filename), 'rU')
    image_filenames = []
    for line in f:
        image_id, image_label = line.rstrip('\n').split(' ')
        if not converted_to_tfrecord:
            image_content = tf.read_file(os.path.join(directory, image_id + '.jpg'))
            image = tf.image.decode_jpeg(image_content)
            resized_image = tf.cast(tf.image.resize_images(image, [250, 250]), tf.uint8)
            image_bytes = sess.run(resized_image).tobytes()
            example = tf.train.Example(features = tf.train.Features(feature = {
                'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [int(image_label) - 1])),
                'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_bytes]))
            }))
            writer = tf.python_io.TFRecordWriter(os.path.join(directory, image_id + '.tfrecord'))
            writer.write(example.SerializeToString())
            writer.close()
            print('generated ' + image_id + '.tfrecord')
        image_filenames.append(os.path.join(directory, image_id + '.tfrecord'))
    f.close()
    return image_filenames


#restore from checkpoint or init variables
sess = tf.Session()
converted_to_tfrecord = False
ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
if ckpt and ckpt.model_checkpoint_path:
    converted_to_tfrecord = True

#convert image to tfrecord and read
images_filename = read_image_label_list_file_and_convert_image_to_tfrecord(os.path.join('ic-data', 'train'), 'train.label')
filename_queue = tf.train.string_input_producer(
    images_filename,
    shuffle = False
    )
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    })
image_raw = tf.decode_raw(features['image'], tf.uint8)
image = tf.reshape(image_raw, [250, 250, 3])
image = tf.cast(image, tf.float32) / 255
label = tf.cast(features['label'], tf.int32)
min_after_dequeue = 500
batch_size = 3
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [image, label], batch_size = batch_size, capacity = capacity,
    min_after_dequeue = min_after_dequeue)
conv2d_layer_one = tf.contrib.layers.conv2d(
    inputs = image_batch,
    num_outputs = 64,
    kernel_size = (7, 7),
    stride = (2, 2),
    padding = 'SAME',
    trainable = True)
pool_layer_one = tf.contrib.layers.max_pool2d(
    inputs = conv2d_layer_one,
    kernel_size = [2, 2],
    stride = [2, 2],
    padding = 'SAME')
conv2d_layers = range(32)
for i in range(0, 6):
    if i == 0:
        conv2d_layers[i] = tf.contrib.layers.conv2d(
            inputs = pool_layer_one,
            num_outputs = 64,
            kernel_size = (3, 3),
            stride = (1, 1),
            padding = 'SAME',
            trainable = True)
    else:
        conv2d_layers[i] = tf.contrib.layers.conv2d(
            inputs = conv2d_layers[i - 1],
            num_outputs = 64,
            kernel_size = (3, 3),
            stride = (1, 1),
            padding = 'SAME',
            trainable = True)
    if i % 2 == 1:
        conv2d_layers[i] = conv2d_layers[i] + conv2d_layers[i - 1]
for i in range(6, 14):
    if i == 6:
        conv2d_layers[i] = tf.contrib.layers.conv2d(
            inputs = conv2d_layers[i - 1],
            num_outputs = 128,
            kernel_size = (3, 3),
            stride = (2, 2),
            padding = 'SAME',
            trainable = True)
    else:
        conv2d_layers[i] = tf.contrib.layers.conv2d(
            inputs = conv2d_layers[i - 1],
            num_outputs = 128,
            kernel_size = (3, 3),
            stride = (1, 1),
            padding = 'SAME',
            trainable = True)
    if i > 7 and i % 2 == 1:
        conv2d_layers[i] = conv2d_layers[i] + conv2d_layers[i - 1]
for i in range(14, 26):
    if i == 14:
        conv2d_layers[i] = tf.contrib.layers.conv2d(
            inputs = conv2d_layers[i - 1],
            num_outputs = 256,
            kernel_size = (3, 3),
            stride = (2, 2),
            padding = 'SAME',
            trainable = True)
    else:
        conv2d_layers[i] = tf.contrib.layers.conv2d(
            inputs = conv2d_layers[i - 1],
            num_outputs = 256,
            kernel_size = (3, 3),
            stride = (1, 1),
            padding = 'SAME',
            trainable = True)
    if i > 15 and i % 2 == 1:
        conv2d_layers[i] = conv2d_layers[i] + conv2d_layers[i - 1]
for i in range(26, 32):
    if i == 26:
        conv2d_layers[i] = tf.contrib.layers.conv2d(
            inputs = conv2d_layers[i - 1],
            num_outputs = 512,
            kernel_size = (3, 3),
            stride = (2, 2),
            padding = 'SAME',
            trainable = True)
    else:
        conv2d_layers[i] = tf.contrib.layers.conv2d(
            inputs = conv2d_layers[i - 1],
            num_outputs = 512,
            kernel_size = (3, 3),
            stride = (1, 1),
            padding = 'SAME',
            trainable = True)
    if i > 27 and i % 2 == 1:
        conv2d_layers[i] = conv2d_layers[i] + conv2d_layers[i - 1]
pool_layer_five = tf.contrib.layers.avg_pool2d(
    inputs = conv2d_layers[31],
    kernel_size = [2, 2],
    stride = [2, 2],
    padding = 'SAME')
flattened_layer_two = tf.reshape(
    pool_layer_five,
    [
        batch_size,
        -1
    ])
final_layer = tf.contrib.layers.fully_connected(
    inputs = flattened_layer_two,
    num_outputs = 12,
    trainable = True)
onehot_labels = tf.one_hot(indices = label_batch, depth = 12)
loss = tf.losses.softmax_cross_entropy(
    onehot_labels = onehot_labels,
    logits = final_layer)
train_op = tf.contrib.layers.optimize_loss(
    loss = loss,
    global_step = tf.contrib.framework.get_global_step(),
    learning_rate = 0.0001,
    optimizer = 'SGD')

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
