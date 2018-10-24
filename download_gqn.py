import os
import sys

import collections


DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)


DATASETS_INFO = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)


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
