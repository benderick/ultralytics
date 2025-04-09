# ultralytics目前接受两种数据布局
## 第一种 images-labels
支持目录和文件的数据划分，图像和标注必须分别位于`images`和`labels`目录中。
 VisDrone
├──  split
│   ├──  test-challenge.txt
│   ├──  test-dev.txt
│   ├──  train.txt
│   └──  val.txt
├──  VisDrone2019-DET-test-challenge
│   ├──  images
│   └──  labels
├──  VisDrone2019-DET-test-dev
│   ├──  images
│   └──  labels
├──  VisDrone2019-DET-train
│   ├──  images
│   │   ├───  0000360_07253_d_0000750.jpg 
│   │   ├─── ...
│   └──  labels
│       ├───  0000360_07253_d_0000750.txt 
│       ├─── ...
├──  VisDrone2019-DET-val
│   ├──  images
│   └──  labels
└──  VisDrone.yaml

**VisDrone.yaml 文件式划分**
```yaml
train: ./split/train.txt  # 4K images
val: ./split/val.txt  # 700 imageS
test: ./split/test-dev.txt  # 20288 of 40670 images, submit to 
nc: 10
names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
```
**train.txt**
```plain
VisDrone/VisDrone2019-DET-train/images/0000002_00005_d_0000014_masked.jpg
VisDrone/VisDrone2019-DET-train/images/0000002_00448_d_0000015_masked.jpg
```
**VisDrone.yaml 目录式划分**
```yaml
path: ./VisDrone # dataset root dir
train: VisDrone2019-DET-train/images # train images (relative to 'path')  6471 images
val: VisDrone2019-DET-val/images # val images (relative to 'path')  548 images
test: VisDrone2019-DET-test-dev/images # test images (optional)  1610 images
# Classes
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

## 第二种 images-images
支持目录和文件的数据划分，图像和对应标注必须位于相同目录中。目录名不要求。
 UAVDT
├──  split
│   ├──  test-challenge.txt
│   ├──  test-dev.txt
│   ├──  train.txt
│   └──  val.txt
├──  UAV-benchmark-M
│   ├──  M0101
│   │   ├──  img000404_masked.jpg
│   │   ├──  img000404_masked.txt
│   │   └── ...
│   ├──  M0201
│   ├──  M0202
│   ├──  M0203
│   └── ... 
└──  UAVDT.yaml 

**UAVDT.yaml**
```yaml
train: ./split/train_ds.txt  # train_ds.txt
val: ./split/valid.txt  # 1.5k images
test: ./split/test.txt  # 
nc: 3
names: ['car', 'truck', 'bus']
```
**train.txt**
```plain
UAVDT/UAV-benchmark-M/M0204/img000006.jpg
UAVDT/UAV-benchmark-M/M0204/img000016.jpg
UAVDT/UAV-benchmark-M/M0204/img000026.jpg
UAVDT/UAV-benchmark-M/M0204/img000036.jpg
UAVDT/UAV-benchmark-M/M0204/img000046.jpg
UAVDT/UAV-benchmark-M/M0204/img000056.jpg
```
## 目录划分和文件划分
目录划分适用于数据已经划分为不同目的文件夹，文件划分适用于数据没有划分为不同目的文件夹。
