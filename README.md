# gqn_datasets_translator

### Data downloader and data converter for DeepMind GQN dataset https://github.com/deepmind/gqn-datasets to use with other libraries than TensorFlow

Don't hesitate to make a pull request. 

**Dependencies**

You need to install:

- TensorFlow [here](https://www.tensorflow.org/install/)
- gsutil [here](https://cloud.google.com/storage/docs/gsutil_install) *Note that gsutil works in python 2.\* only*


**Download the tfrecord dataset**

If you want to download the entire dataset:
```shell
gsutil -m cp -R gs://gqn-dataset/<dataset> .
```

If you want to download a proportion of the dataset only:
```shell
python download_gqn.py <dataset> <proportion>
```

**Convert the raw dataset**

Command line options:
```shell
usage: convert2file.py [-h] [-b BATCH_SIZE] [-n FIRST_N] [-m MODE]
                       base_dir dataset

Convert gqn tfrecords to gzip files.

positional arguments:
  base_dir              base directory of gqn dataset
  dataset               datasets to convert, eg. shepard_metzler_5_parts

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        number of sequences in each output file
  -n FIRST_N, --first-n FIRST_N
                        convert only the first n tfrecords if given
  -m MODE, --mode MODE  whether to convert train or test
```

Convert all records with all sequences in sm5 train (400 records, 2000 seq each):
```shell
python convert2file.py ~/gqn_dataset shepard_metzler_5_parts
```

Convert first 20 records with batch size of 128 in sm5 test:
```shell
python convert2file.py ~/gqn_dataset shepard_metzler_5_parts -n 20 -b 128 -m test
```

**Size of the datasets:**

| Names        | Sizes           |
| ------------- |:-------------:|
| _total_ | 1.45 Tb |
| ------------- | --------------|
| jaco      | 198.97 Gb |
| mazes      | 136.23 Gb |
| rooms\_free\_camera\_no\_object\_rotations | 255.75 Gb |
| rooms\_free\_camera\_with\_object\_rotations | 598.75 Gb |
| rooms\_ring\_camera | 250.89 Gb |
| shepard\_metzler\_5\_parts | 21.09 Gb |
| shepard\_metzler\_7\_parts | 23.68 Gb |
