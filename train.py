import tensorflow as tf

def read_image_label_list_file_and_convert_image_to_tfrecord(directory, filename):
    global sess

    f = open(directory + '/' + filename, 'r')
    image_filenames = []
    image_labels = []
    for line in f:
        image_id, image_label = line[:-2].split(' ')
        image_content = tf.read_file(directory + '/' + image_id + '.jpg')
        image = tf.image.decode_jpeg(image_content)
        resized_image = tf.image.resize_images(image, [250, 250])
        image_bytes = sess.run(tf.cast(resized_image, tf.int32)).tobytes()
        example = tf.train.Example(features = tf.train.Features(feature = {
            'label': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_label])),
            'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_bytes]))
        }))
        writer = tf.python_io.TFRecordWriter(directory + '/' + image_id + '.tfrecord')
        writer.write(example.SerializeToString())
        writer.close()
        print image_id + '.tfrecord'
        image_filenames.append(directory + '/' + image_id + '.tfrecord')
        image_labels.append(int(image_label))
    return image_filenames, image_labels

def read_images_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents)
    return example, label

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord)

image_list, label_list = read_image_label_list_file_and_convert_image_to_tfrecord('ic-data/train', 'train.label')
images = tf.convert_to_tensor(image_list, dtype = tf.string)
labels = tf.convert_to_tensor(label_list, dtype = tf.int32)
input_queue = tf.train.slice_input_producer([images, labels])
image, label = read_images_from_disk(input_queue)

coord.request_stop()
coord.join(threads)
