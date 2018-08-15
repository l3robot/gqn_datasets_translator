import os
import sys

from convert2torch import _DATASETS as DATASETS_INFO

if  len(sys.argv) < 3:
    print(' [!] you need to give a <str> dataset and a <float> proportion to download')
    exit()


PROP = float(sys.argv[2])
DATASET = sys.argv[1]
dataset_info = DATASETS_INFO[DATASET]

top_path = f'{DATASET}'
train_path = f'{DATASET}/train'
test_path = f'{DATASET}/test'

train_nb = int(PROP * dataset_info.train_size)
test_nb = int(PROP * dataset_info.test_size)

train_length = len(str(dataset_info.train_size))
train_template = '{:0%d}-of-{:0%d}.tfrecord' % (train_length, train_length)

test_length = len(str(dataset_info.test_size))
test_template = '{:0%d}-of-{:0%d}.tfrecord' % (test_length, test_length)

os.mkdir(top_path)
os.mkdir(train_path)
os.mkdir(test_path)

header = 'gsutil -m cp gs://gqn-dataset/{}'.format(DATASET)

## train copy
for i in range(train_nb):
    file = train_template.format(i+1, dataset_info.train_size)
    command = '{0}/train/{1} {2}/{1}'.format(header, file, train_path)
    # print(command)
    os.system(command)

## test copy
for i in range(test_nb):
    file = test_template.format(i+1, dataset_info.test_size)
    command = '{0}/test/{1} {2}/{1}'.format(header, file, test_path)
    # print(command)
    os.system(command)
