# Melanoma Recognition via Visual Attention  

***
Updates:
* Jan 2020: The code is upgraded to support PyTorch >= 1.1.0. If you are using an older version, be sure to read the following warnings.

***

**WARNING** If you are using PyTorch < 1.1.0, pay attention to the following two points.
* You should adjust learning rate before calling optimizer.step(). Reference: [how to adjust learning rate](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).

>Prior to PyTorch 1.1.0, the learning rate scheduler was expected to be called before the optimizer’s update; 1.1.0 changed this behavior in a BC-breaking way. If you use the learning rate scheduler (calling scheduler.step()) before the optimizer’s update (calling optimizer.step()), this will skip the first value of the learning rate schedule. If you are unable to reproduce results after upgrading to PyTorch 1.1.0, please check if you are calling scheduler.step() at the wrong time.

* Prior to 1.1, Tensorboard is not natively supported in PyTorch. An alternative is to use [tensorboardX](https://github.com/lanpa/tensorboardX). Good news is the APIs are the same (in fact torch.utils.tensorboard === tensorboardX).  

***

If you use the code for your own reasearch, please cite the following paper :)

@inproceedings{yan2019melanoma,  
  title={Melanoma Recognition via Visual Attention},  
  author={Yan, Yiqi and Kawahara, Jeremy and Hamarneh, Ghassan},  
  booktitle={International Conference on Information Processing in Medical Imaging},  
  pages={793--804},  
  year={2019},  
  organization={Springer}  
}  

[Project webpage](https://saoyan.github.io/posts/2019/03/07)   

<img src="https://github.com/SaoYan/Attention-Skin/blob/master/assets/network.png" alt="network" width="500">  

<img src="https://github.com/SaoYan/Attention-Skin/blob/master/assets/atten.jpg" alt="visualization" width="500">  

## Pre-traind models

[Google drive link](https://drive.google.com/open?id=1dwnpHfTpy-zSe3jybPOmELxs51iQF1mG)  

## How to run

### 1. Dependences  

* PyTorch >= 1.1.0  
* torchvision  
* scikit-learn  

### 2. Data preparation  

ISIC 2016 [download here](https://challenge.kitware.com/#challenge/560d7856cad3a57cfde481ba); organize the data as follows  

* data_2016/  
  * Train/  
    * benign/  
    * malignant/  
  * Test/  
    * benign/  
    * malignant/  

ISIC 2017 [download here](https://challenge.kitware.com/#challenge/n/ISIC_2017%3A_Skin_Lesion_Analysis_Towards_Melanoma_Detection); organize the data as follows  

* data_2017/  
  * Train/  
    * melanoma/  
    * nevus/
    * seborrheic_keratosis/  
  * Val/  
    * melanoma/  
    * nevus/
    * seborrheic_keratosis/   
  * Test/  
    * melanoma/  
    * nevus/
    * seborrheic_keratosis/   
  * Train_Lesion/  
    * melanoma/  
    * nevus/
    * seborrheic_keratosis/   
  * Train_Dermo/  
    * melanoma/  
    * nevus/
    * seborrheic_keratosis/   

Under the folder *Train_Lesion* is the lesion segmentation map (ISIC2017 part I); under the folder *Train_Dermo* is the map of dermoscopic features (ISIC2017 part II). The raw data of dermoscopic features require some preprocessing in order to convert to binary maps. What is under Train_Dermo is the union map of four dermoscopic features. Note that not all of the images have dermoscopic features (i.e, some of the maps are all zero).  

### 3. Training

1. Training without any attention map regularization (with only the classification loss, i.e, *AttnMel-CNN* in the paper):  

* train on ISIC 2016  

```
python train.py --dataset ISIC2016 --preprocess --over_sample --focal_loss --log_images
```

* train on ISIC 2017 (by default)  

```
python train.py --dataset ISIC2017 --preprocess --over_sample --focal_loss --log_images
```

2. Training with attention map regularization (*AttnMel-CNN-Lesion* or *AttnMel-CNN-Dermo* in the paper):  

We only train on ISIC2017 for these two models.  

* *AttnMel-CNN-Lesion* (by default)  

```
python train_seg.py --seg lesion --preprocess --over_sample --focal_loss --log_images
```

* *AttnMel-CNN-Dermo*  

```
python train_seg.py --seg dermo --preprocess --over_sample --focal_loss --log_images
```

3. Testing

```
python test.py --dataset ISIC2016  
```

or

```
python test.py --dataset ISIC2017  
```

## LICENSE  

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
