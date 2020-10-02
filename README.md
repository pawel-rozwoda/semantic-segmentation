# Semantic segmentation solution for 
# https://www.kaggle.com/c/understanding_cloud_organization

## The purpose of following solution is to perform semantic segmentation on cloud sattelite images  

## ML model is Unet got from here:
### https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/

## 1. Data preprocessing
### Crop high-res jpeg file into smaller ones(i.e 140x140)
### Generate true label masks from csv file (4 channel binary mask for 4 classes)

## 2. Test
### `combine_images.py` file perform cropping test

## 3. Metrics (losses.ipynb)
### Loss I used is SoftDiceLoss(X, y) + BCE(X, y)
### Results of optimization can be found in losses.ipynb

## 4. Data
### a) get data from here  https://www.kaggle.com/c/understanding_cloud_organization/data
### b) create data directory
### c) unzip to data/train_images directory
### d) execute `python gen_data.py`. this would generate small images as well as new csv mask file

## 5. Train
### Execute `python train_data.py`. In default 10 epochs


## 6. Model prediction
### Execute `predict.py` 

