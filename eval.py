infer_file = open('result.list', 'rU')
real_file = open('extra.label', 'rU')
infer_num = 0
real_num = 0
right_num = 0

real_labels = {}
labels_sum = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0, '12':0}
labels_right_num = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0, '10':0, '11':0, '12':0}

for line in real_file:
    image_id, image_label = line.rstrip('\n').split(' ')
    real_labels[image_id] = image_label
    real_num = real_num + 1
    labels_sum[image_label] = labels_sum[image_label] + 1
real_file.close()

for line in infer_file:
    image_id, image_label = line.rstrip('\n').split(' ')
    infer_num = infer_num + 1
    if real_labels[image_id] == image_label:
        right_num = right_num + 1
        labels_right_num[image_label] = labels_right_num[image_label] + 1
infer_file.close

print('records number in extra.label: %d' % real_num)
print('records number in result.list: %d' % infer_num)
print('accurate rate: %f' % (right_num * 1. / infer_num))
for i in range(1, 13):
    print('accurate rate for label %d: %f' % (i, labels_right_num['%d' % i] * 1. / labels_sum['%d' % i]))
