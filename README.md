# Torch Implementation for Stacked Attention Networks for Image Question Answering 

## [!] Code Fixing Now

Train a Stacked Attention Network for Image Question Answering on VQA dataset.  For more information, please refer the [paper](https://arxiv.org/abs/1511.02274) and original [theano code](https://github.com/zcyang/imageqa-san).

![teaser results](https://github.com/JamesChuanggg/san-torch/blob/master/file/VQA.png?raw=true)

### Requirements
This code is written in Lua and requires [Torch](http://torch.ch/). The preprocssinng code is in Python, and you need to install [NLTK](http://www.nltk.org/) if you want to use NLTK to tokenize the question.

You also need to install the following package in order to sucessfully run the code.

- [cudnn.torch](https://github.com/soumith/cudnn.torch)
- [torch-hdf5](https://github.com/deepmind/torch-hdf5)
- [lua-cjson](http://www.kyne.com.au/~mark/software/lua-cjson.php)
- [loadcaffe](https://github.com/szagoruyko/loadcaffe)
- [iTorch](https://github.com/facebook/iTorch)

##### Download Dataset
We simply follow the steps provide by [HieCoAttenVQA](https://github.com/jiasenlu/HieCoAttenVQA) to prepare VQA data.
The first thing you need to do is to download the data and do some preprocessing. Head over to the `data/` folder and run

```
$ python vqa_preprocessing.py --download 1 --split 1
```
`--download Ture` means you choose to download the VQA data from the [VQA website](http://www.visualqa.org/) and `--split 1` means you use COCO train set to train and validation set to evaluation. `--split 2 ` means you use COCO train+val set to train and test set to evaluate. After this step, it will generate two files under the `data` folder. `vqa_raw_train.json` and `vqa_raw_test.json`

##### Download Image Model
Here we use VGG_ILSVRC_19_layers [model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77) and Deep Residual network implement by Facebook [model](https://github.com/facebook/fb.resnet.torch). 

##### Generate Image/Question Features

Head over to the `prepro` folder and run
```
$ python prepro_vqa.py --input_train_json ../data/vqa_raw_train.json --input_test_json ../data/vqa_raw_test.json --num_ans 1000
```
to get the question features. --num_ans specifiy how many top answers you want to use during training. You will also see some question and answer statistics in the terminal output. This will generate two files in `data/` folder, `vqa_data_prepro.h5` and `vqa_data_prepro.json`.

Then we are ready to extract the image features by VGG 19.

```
$ th prepro_img_vgg.lua -input_json ../data/vqa_data_prepro.json -image_root /home/jiasenlu/data/ -cnn_proto ../image_model/VGG_ILSVRC_19_layers_deploy.prototxt -cnn_model ../image_model/VGG_ILSVRC_19_layers.caffemodel
```
you can change the `-gpuid`, `-backend` and `-batch_size` based on your gpu.

##### Train the model

We have everything ready to train the VQA. Back to the `main` folder

```
th train.lua
```

##### Evaluate the model
In main folder run 
```
th eval.lua -start_from [model path]
```
