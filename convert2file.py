import os
import tensorflow as tf
import torch
import gzip
from functools import partial
from collections import namedtuple
import argparse as ap
import multiprocessing as mp


"""
This file converts tfrecords in deepmind gqn dataset to gzip files. Each tfrecord will be converted
to a single gzip file (561-of-900.tfrecord -> 561-of-900.pt.gz).

Each gzip file contains a list of tuples, where each tuple is of (images, poses)
For example, when converting the shepard_metzler_5_parts dataset with batch_size of 32, the gzip 
file contains a list of length 32, each tuple contains images (15,64,64,3) and poses (15,5), where
15 is the sequence length.

In the original implementation, each sequence is converted to a gzip file, this results in more than
800K small files on the disk. Here we choose to pack multiple sequences into one gzip file, thus 
avoiding having too many small files. Note that the gqn implementation from wohlert 
(https://github.com/wohlert/generative-query-network-pytorch) works with the original version. In 
order for it to work with the new format, one can simply change (in wohlert gqn) batch_size to 1 
and do a squeeze after the loader.

It is also recommended to remove the first 500 records of both shepard metzler dataset as they 
only contain 20 sequences, compared to the last 400 records which contain 2000 sequences.

Example:
convert all records with all sequences in sm5 train (400 records, 2000 seq each)
python convert2file.py ~/gqn_dataset shepard_metzler_5_parts 

Convert first 20 records with batch size of 128 in sm5 test
python convert2file.py ~/gqn_dataset shepard_metzler_5_parts -n 20 -b 128 -m test
"""

tf.logging.set_verbosity(tf.logging.ERROR)  # disable annoying logging
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable gpu

DatasetInfo = namedtuple('DatasetInfo', ['image_size', 'seq_length'])

all_datasets = dict(
    jaco=DatasetInfo(image_size=64, seq_length=11),
    mazes=DatasetInfo(image_size=84, seq_length=300),
    rooms_free_camera_with_object_rotations=DatasetInfo(image_size=128, seq_length=10),
    rooms_ring_camera=DatasetInfo(image_size=64, seq_length=10),
    rooms_free_camera_no_object_rotations=DatasetInfo(image_size=64, seq_length=10),
    shepard_metzler_5_parts=DatasetInfo(image_size=64, seq_length=15),
    shepard_metzler_7_parts=DatasetInfo(image_size=64, seq_length=15)
)

_pose_dim = 5


def collect_files(path, ext=None, key=None):
    if key is None:
        files = sorted(os.listdir(path))
    else:
        files = sorted(os.listdir(path), key=key)

    if ext is not None:
        files = [f for f in files if os.path.splitext(f)[-1] == ext]

    return [os.path.join(path, fname) for fname in files]


def convert_record(record, info, batch_size=None):
    print(record)

    path, filename = os.path.split(record)
    basename = os.path.splitext(filename)[0]
    scenes = process_record(record, info, batch_size)
    # scenes is a list of tuples (image_seq, pose_seq)
    out = os.path.join(path, f'{basename}.pt.gz')
    save_to_disk(scenes, out)


def save_to_disk(scenes, path):
    with gzip.open(path, 'wb') as f:
        torch.save(scenes, f)


def process_record(record, info, batch_size=None):
    engine = tf.python_io.tf_record_iterator(record)

    scenes = []
    for i, data in enumerate(engine):
        if i == batch_size:
            break
        scene = convert_to_numpy(data, info)
        scenes.append(scene)

    return scenes


def process_images(example, seq_length, image_size):
    """Instantiates the ops used to preprocess the frames data."""
    images = tf.concat(example['frames'], axis=0)
    images = tf.map_fn(tf.image.decode_jpeg, tf.reshape(images, [-1]),
                       dtype=tf.uint8, back_prop=False)
    shape = (image_size, image_size, 3)
    images = tf.reshape(images, (-1, seq_length) + shape)
    return images


def process_poses(example, seq_length):
    """Instantiates the ops used to preprocess the cameras data."""
    poses = example['cameras']
    poses = tf.reshape(poses, (-1, seq_length, _pose_dim))
    return poses


def convert_to_numpy(raw_data, info):
    seq_length = info.seq_length
    image_size = info.image_size

    feature = {'frames': tf.FixedLenFeature(shape=seq_length, dtype=tf.string),
               'cameras': tf.FixedLenFeature(shape=seq_length * _pose_dim, dtype=tf.float32)}
    example = tf.parse_single_example(raw_data, feature)

    images = process_images(example, seq_length, image_size)
    poses = process_poses(example, seq_length)

    return images.numpy().squeeze(), poses.numpy().squeeze()


if __name__ == '__main__':
    tf.enable_eager_execution()
    parser = ap.ArgumentParser(description='Convert gqn tfrecords to gzip files.')
    parser.add_argument('base_dir', nargs=1,
                        help='base directory of gqn dataset')
    parser.add_argument('dataset', nargs=1,
                        help='datasets to convert, eg. shepard_metzler_5_parts')
    parser.add_argument('-b', '--batch-size', type=int, default=None,
                        help='number of sequences in each output file')
    parser.add_argument('-n', '--first-n', type=int, default=None,
                        help='convert only the first n tfrecords if given')
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='whether to convert train or test')
    args = parser.parse_args()

    base_dir = os.path.expanduser(args.base_dir[0])
    dataset = args.dataset[0]

    print(f'base_dir: {base_dir}')
    print(f'dataset: {dataset}')

    info = all_datasets[dataset]
    data_dir = os.path.join(base_dir, dataset)
    records = collect_files(os.path.join(data_dir, args.mode), '.tfrecord')

    if args.first_n is not None:
        records = records[:args.first_n]

    num_proc = mp.cpu_count()
    print(f'converting {len(records)} records in {dataset}/{args.mode}, with {num_proc} processes')

    with mp.Pool(processes=num_proc) as pool:
        f = partial(convert_record, info=info, batch_size=args.batch_size)
        pool.map(f, records)

    print('Done')