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
