import tensorflow as tf

def read_image_label_list_file(directory, filename):
    f = open(directory + '/' + filename, 'r')
    image_filenames = []
    image_labels = []
    for line in f:
        image_id, image_label = line[:-2].split(' ')
        image_filenames.append(directory + '/' + image_id + '.jpg')
        image_labels.append(int(image_label))
    return image_filenames, image_labels

def read_images_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents)
    return example, label


image_list, label_list = read_image_label_list_file('ic-data/train', 'train.label')
images = tf.convert_to_tensor(image_list, dtype = tf.string)
labels = tf.convert_to_tensor(label_list, dtype = tf.int32)
input_queue = tf.train.slice_input_producer([images, labels])
image, label = read_images_from_disk(input_queue)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)

    print sess.run(image)

    coord.request_stop()
    coord.join(threads)
