import os
import tensorflow as tf

def read_image_label_list_file_and_convert_image_to_tfrecord(directory, filename):
    global sess
    global converted_to_tfrecord
    global saver

    f = open(os.path.join(directory, filename), 'r')
    image_filenames = []
    image_labels = []
    for line in f:
        image_id, image_label = line[:-2].split(' ')
        if not sess.run(converted_to_tfrecord):
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
    if not sess.run(converted_to_tfrecord):
        sess.run(converted_to_tfrecord.assign(True))
        saver.save(sess, 'train.ckpt')
    return image_filenames

sess = tf.Session()

converted_to_tfrecord = tf.Variable(False, name = 'converted_to_tfrecord')
saver = tf.train.Saver()

#restore from checkpoint or init variables
ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('restored from %s' % ckpt.model_checkpoint_path)
else:
    init_variables = tf.global_variables_initializer()
    sess.run(init_variables)

image_list = read_image_label_list_file_and_convert_image_to_tfrecord(os.path.join('ic-data', 'train'), 'train.label')

images = tf.convert_to_tensor(image_list, dtype = tf.string)

sess.close()
