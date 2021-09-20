# Face alignment in Full Pose Range: A 3D Total Solution
## Original repo: [3DDFA](https://github.com/cleardusk/3DDFA)

**\[Todo\]**
- [x] Online random crop augmentation.
- [ ] Replace basel face model to newest version, current version is 2009.

## Prerequisites
#### 1. Linear Algebra
- Basis of vector space.
- Camera matrix: Rotation matrix, translation vector, scale.
- Euler angles.
- Recommend resources: [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

#### 2. Statistics:
- PCA.
- Recommend resources: [StatQuest](https://www.youtube.com/watch?v=FgakZw6K1QQ&t=1069s&ab_channel=StatQuestwithJoshStarmer)

#### 3. 3D Morphable Model (3DMM):
- Recommend resources: [What is a Linear 3D Morphable Face Model?](https://www.youtube.com/watch?v=MlGkzFeyCYc&ab_channel=KalleBladin); [BFM](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-0&id=basel_face_model); [FLAME](https://flame.is.tue.mpg.de/)

## Dataset
#### 1. General:
- Input image: Cropped face image.
- Label: 3DMM parameters; created from flatten camera matrix, flatten shape and flatten expression parameters; in `.mat` format.

#### 2. Public dataset:
|Name   |Raw data   |Processed   |Note   |
|:---:|:---:|:---:|:---:|
|300WLP|[Link](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)|[Link](https://drive.google.com/file/d/1XVUYvMhLOvz4EquN1O_8z6gjnqT6p_7o/view?usp=sharing)   |Training|
|300VW|[Link](https://drive.google.com/drive/folders/1MDrs4Hh1oMm1uWL3r64MpSpNIg55v5xl?usp=sharing)|[Link](https://drive.google.com/file/d/1XVUYvMhLOvz4EquN1O_8z6gjnqT6p_7o/view?usp=sharing)   |Training|
|AFLW2000-3D|[Link](https://drive.google.com/file/d/1SaAnw9HTcUKcogkFOg3dbjGOJ9Edx73D/view?usp=sharing)|[Link](https://drive.google.com/file/d/1XVUYvMhLOvz4EquN1O_8z6gjnqT6p_7o/view?usp=sharing)   |Validation|

#### 3. Dataset description:
- The end purpose of the model is predicting accurately 68 3D landmarks given a face image. The condition of the face could be:
    - Large pose: Pitch in range (-40, 40), yaw in range (-70, 70), roll in range (-180, 180).
    - Closed and opened eyes.

- The training dataset consist of :
    - `300WLP_3ddfa`: Face with high pitch and yaw value.
    - `300VW_closed_eyes_3ddfa`: Face with closed eyes in both original and medium pose range. Medium pose face is synthesized from original face image using basel face model. The medium pose file name is in format \*_1.jpg.
    - `300VW_opened_eyes_3ddfa`: Face with opened eyes in both original and large pose range. Large pose face is synthesized from original face image using basel face model. The medium pose file name is in format \*_1.jpg.

- Augmentation:
    - Vanilla augmentation using imgaug. Heavy augment is not preferred because it might destroy the image.
    - Hide face: A part of the face that contain landmarks is intentionally hiding using objects. Currently face is covered by black box and hands.
    - Inplane rotation: Rotate the image inplane in range (-180, 180).

- Generating data:
    - Most of public datasets do not directly contain the label for the 3DDFA model. To generate the label for 3DDFA, firstly the sample need to have it corresponding **68 3D face landmarks**.
    - When the 3D face landmarks are obtained, the next step is fitting 3D landmarks to get the 3DMM parameters. Please refer to this [repo](https://github.com/dd-dos/face3d) to generate data. Some fitting examples have been already provided.

- Training ready:
    - Each dataset have to be prepared in the following structure:
        ```bat
        dataset1
            |--img1.jpg
            |--params1.mat
            |--img2.jpg
            |--params2.mat
            ...
        dataset2
            |--img1.jpg
            |--params1.mat
            |--img2.jpg
            |--params2.mat
            ...
        ```
    - Multiple dataset is supported.
    
## Training
1. Requirements: `pip install -r requirements.txt`
2. Place your dataset at `data`, you have to `mkdir` yourself.
3. Reconfig the training script at `script/train.sh`.
4. `sh script/train.sh`

## Inference
1. Reconfig the testing script at `script/test.sh` to contain your model path.
2. `sh script/test.sh`

## Benchmark
|Backbone   |Type   |Device       |In-memory size   |Inference speed (full flow)   |
|:---:|:---:|:---:|:---:|:---:|
|mobilenet_v1|mobilenet_1|11th Gen Intel(R) Core(TM) i7-1165G7 |N/A|50 fps|
|mobilenet_v2|mobilenet_2|11th Gen Intel(R) Core(TM) i7-1165G7 |756 Mb|25 fps|

## Demo
[Imgur](https://imgur.com/a/8N7n727)

## Other resources
1. Hand for 



















