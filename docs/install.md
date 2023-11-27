# Installation Guide 
## Prerequisites
### Pose

Install MMPose. Follow instruction from [MMPose](https://mmpose.readthedocs.io/en/latest/installation.html).

### Mesh
Install required modules for FrankMocap

```shell
pip install chumpy
pip install scipy
pip install smplx
pip install opencv-python
```

## Installation Steps
#### 1. Navigate to your Project Folder
First, navigate to the project folder:
```shell
cd myproject
```

#### 2. Clone the Repository & Move the Source Folder
Open a terminal or command prompt and run the following command to clone the repository into your desired project folder:
```shell
git clone https://github.com/seoul-dynamics/vimos.git
```

Then Move the Source Folder to Project Folder
```shell
mv vimos vimos_temp
mv vimos_temp/vimos .
rm -rf vimos_temp
```

## [Optional] Setup FrankMocap if you Want to Use Mesh Models

#### 1. Setting SMPL Models.Setting SMPL/SMPL-X Models
We use SMPL and SMPL-X model as 3D pose estimation output. You have to download them from the original website.
Download SMPL Model (Neutral model: basicModel_neutral_lbs_10_207_0_v1.0.0.pkl):    
    - Download in the original [website](http://smplify.is.tue.mpg.de/login). You need to register to download the SMPL data.
    - Put the file as: 'vimos/model/mesh/extra_data/smpl.pkl'

#### 2. Download Models
Run following script to download the models, in your project folder. Please be careful that the script is in vimos/scripts **not** vimos/vimos. If you removed original repository folder in 2, you should re-download the script.
```shell
sh vimos/scripts/download_models.sh
```

