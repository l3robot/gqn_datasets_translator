# gqn_datasets_translator

### Data downloader and data converter for DeepMind GQN dataset https://github.com/deepmind/gqn-datasets to use with other libraries than TensorFlow

Don't hesitate to make a pull request. 

**Usage:**

If you want to download the entire dataset:

```shell
gsutil -m cp -R gs://gqn-dataset/<dataset> .
python convert2torch.py <dataset>
```

If you want to download a proportion of the dataset only:

```shell
python download_gqn.py <dataset> <proportion>
python convert2torch.py <dataset>
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
