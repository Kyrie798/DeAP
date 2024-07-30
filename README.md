# DeAP: Learning Degradation-aware Prior for Image Deblurring

#### News
- **July 30, 2024:** Training and testing codes are released :fire:

<hr />

> **Abstract:** *While image deblurring has made significant progress recently, existing methods still suffer from a challenge in handling complex and diverse blur images caused by camera shake and object movement. Inspired by a fact that different images are degraded in dissimilar ways, we propose to explore image-level degradation knowledge to perform deblurring for complex blur images. We thereby introduce a novel framework to learn degradation-aware prior for image deblurring, termed DeAP. The framework incorporates a momentum contrast feature module (MCFM) to model image-level degradation-aware knowledge as visual priors. The resulting visual priors are then embedded into model's hierarchical features in a unified manner, providing multi-scale degradation-aware knowledge. Benefiting from rich degradation knowledge being modeled, our DeAP framework effectively enhances deblurring ability for complex and diverse blur images. Extensive experiments are conducted on three widely-adopted datasets, including GoPro, HIDE, and RealBlur. The experimental results confirm that our DeAP achieves state-of-the-art performance (e.g., 33.82 dB in PSNR on GoPro dataset).* 
<hr />

## Installation
```
cd DeAP
conda create -n DeAP python=3.9.19
source activate DeAP
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cudatoolkit=11.7 -c pytorch -c conda-forge
pip install opencv-python tqdm glog scikit-image albumentations
pip install -U albumentations[imgaug]
pip install albumentations==1.1.0
```
## Network Architecture
<img src="./Figure/DeAP.png"/>

## Training and Testing
### 1. Data Preparation
Download "[GoPro](https://drive.google.com/drive/folders/1bEZO-l6sI9NXMRd98ldi74kCGAnw4bLQ)" dataset into './datasets' </br>
For example: './datasets/GoPro'
### 2. Training
* The training script uses 8 GPUs by default.
* Run the following command
```
sh run.sh
```
### 3. Testing
**For testing on GoPro/HIDE dataset** </br>
* You should modify the image path in train_config.py. </br>
* Run the following command
```
python tools/predict_GoPro.py
```
**For testing on RealBlur-J/RealBlur-R dataset** </br>
* You should modify the image path in train_config.py. </br>
* Run the following command
```
python tools/predict_RealBlur.py
```
## Evaluation
Before you evaluate, you should download the results into './out'.
* For evaluation on GoPro/HIDE results in MATLAB, Run the following command
```
evaluation_GoPro.m
```
* For evaluation on RealBlur results, Run the following command
```
python evaluate_RealBlur.py
```
## Acknowledgment
This repo is build upon [Stripformer](https://github.com/pp00704831/Stripformer-ECCV-2022-). We acknowledg these excellent implementations.