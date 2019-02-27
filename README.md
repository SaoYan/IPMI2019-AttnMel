# Melanoma Recognition via Visual Attention  

***
Code for our IPMI paper:  

Yiqi Yan, Jeremy Kawahara, and Ghassan Hamarneh, “Melanoma Recognition via Visual Attention”. To appear in International Conference on Information Processing in Medical Imaging (IPMI), 2019.  

***

![]()  
![]()  

## Pre-traind models

[Google drive link](https://drive.google.com/open?id=1dwnpHfTpy-zSe3jybPOmELxs51iQF1mG)  

## How to run

### 1. Dependences  

* PyTorch (>= 0.4, 1.0 preferred)  
* torchvision  
* tensorboardX  
* Pillow  
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

* data_2016/  
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

Switching between ISIC2016 and ISIC2017 data: modify the code following the comments at the beginning of the training script

> switch between ISIC 2016 and 2017
> modify the following contents:
> 1. import from data_2016/import from data_2017
> 2. num_aug: x2 for ISIC 2017; x5 for ISIC 2016
> 3. root_dir of preprocess_data
> 4. mean and std of transforms.Normalize

Training without any attention map regularization (with only the classification loss, *AttnMel-CNN* in the paper):  

```
python train.py --preprocess --over_sample --focal_loss --log_images
```


Training with attention map regularization (*AttnMel-CNN-Dermo* or *AttnMel-CNN-Lesion* in the paper):  

```
python train_seg.py --preprocess --over_sample --focal_loss --log_images
```

Testing

```
python test.py  
```
