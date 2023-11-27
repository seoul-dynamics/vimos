# Installation Guide 
## Prerequisites
### Pose

Install MMPose. Follow instruction from [MMPose](https://mmpose.readthedocs.io/en/latest/installation.html).

### Mesh
#### 1. Install required modules from FrankMocap

```shell
pip install chumpy
pip install scipy
pip install smplx
pip install opencv-python
```

#### 2. Setting SMPL Models.
- Setting SMPL/SMPL-X Models
    - We use SMPL and SMPL-X model as 3D pose estimation output. You have to download them from the original website.
    - Download SMPL Model (Neutral model: basicModel_neutral_lbs_10_207_0_v1.0.0.pkl):    
        - Download in the original [website](http://smplify.is.tue.mpg.de/login). You need to register to download the SMPL data.
        - Put the file as: './vimos/model/mesh/extra_data/smpl.pkl'

#### 3. Download Models
```shell
sh scripts/download_models.sh
```

## Installation Steps
#### 1. Navigate to your Project Folder
First, navigate to the project folder:
```shell
cd myproject
```

#### 2. Clone the Repository
Open a terminal or command prompt and run the following command to clone the repository into your desired project folder:
```shell
git clone https://github.com/seoul-dynamics/vimos.git
```

