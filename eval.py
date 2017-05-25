infer_file = open('result.list', 'rU')
real_file = open('extra.label', 'rU')
infer_num = 0
real_num = 0
right_num = 0

real_labels = {}
for line in real_file:
    image_id, image_label = line[:-1].split(' ')
    real_labels[image_id] = image_label
    real_num = real_num + 1
real_file.close()

for line in infer_file:
    image_id, image_label = line[:-1].split(' ')
    infer_num = infer_num + 1
    if real_labels[image_id] == image_label:
        right_num = right_num + 1
infer_file.close

print('records number in extra.label: %d' % real_num)
print('records number in result.list: %d' % infer_num)
print('accurate rate: %f' % (right_num * 1. / infer_num))
