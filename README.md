# Diabetic Retinopathy Classifier

This repository aims to classify retina images taken using fundus photography into 5 categories of diabetic retinopathy (DR): No DR, Mild, Moderate, Severe, and Proliferative DR.

# Dataset
Dataset is downloaded from kaggle. The link to the website is [here](https://www.kaggle.com/mikolajbabula/disaster-images-dataset-cnn-model?select=DisasterModel).
It contains 2000 train data and test data.
However, due to the huge size of the original dataset, we only used the training data as our whole dataset. We further split the 3662 images into train and validation data.

<table>
<tr>
<td> <img src="data/train/cyclone/0.jpg" width="100"> <p><em>Cyclone</em> </p></td>
<td><img src="data/train/earthquake/0.jpg" width="100"> <p><em>Earthquake</em> </p></td>
<td> <img src="data/train/flood/0.jpg" width="100"> <p><em>Flood</em> </p></td>
<td> <img src="data/train/wildfire/0.jpg" width="100"> <p><em>Wildfire</em> </p></td>
</tr>
</table>

# How to Run
Prepare `data` folder to be structured like this
    
 
    ├── data
    |    ├── train
    |        ├── cyclone
    |        ├── earthquake
    |        ├── flood
    |        ├── wildfire
    |    ├── validation
    |        ├── cyclone
    |        ├── earthquake
    |        ├── flood
    |        ├── wildfire
    └── ...        
            
`python3 train_disaster.py -h`

```
usage: train.py [-h] -e EPOCH [-b BATCH_SIZE] -opt OPTIMIZER [-t TRANSFER] -l
                LOSS -act ACTIVATION

optional arguments:
  -h, --help            show this help message and exit
  -e EPOCH, --epoch EPOCH
                        training epoch
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        training batch size
  -opt OPTIMIZER, --optimizer OPTIMIZER
                        optimizer for compile model option: sgd, adam,
                        rmsprop, adagrad
  -t TRANSFER, --transfer TRANSFER
                        transfer learning pretrain model, option: vgg16,
                        vgg19, resnet50
  -l LOSS, --loss LOSS  training loss to compile option:
                        categorical_crossentropy, binary_crossentropy
  -act ACTIVATION, --activation ACTIVATION
                        training activation function to compile, option: tanh, relu, sigmoid

```
## Example
With simple architecture

`python3 train.py -e 3 -opt adam -l binary_crossentropy -act relu`

With Transfer learning VGG19

`python3 train.py -e 3 -opt adam -l binary_crossentropy -act relu -t vgg19` 


# To-do
- add readme ✅
- implement transfer learning ✅
- implement hyperparameter tuning ✅
