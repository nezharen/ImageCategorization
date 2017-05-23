import os
import tensorflow as tf

def read_image_label_list_file_and_convert_image_to_tfrecord(directory, filename):
    global sess

    f = open(os.path.join(directory, filename), 'r')
    image_filenames = []
    image_labels = []
    for line in f:
        image_id, image_label = line[:-2].split(' ')
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
        print image_id + '.tfrecord'
        image_filenames.append(os.path.join(directory, image_id + '.tfrecord'))
    return image_filenames

sess = tf.Session()

image_list = read_image_label_list_file_and_convert_image_to_tfrecord(os.path.join('ic-data', 'train'), 'train.label')
images = tf.convert_to_tensor(image_list, dtype = tf.string)

