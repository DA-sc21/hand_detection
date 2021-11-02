# hand_detection

## analysis result

### structure
```
.
├── README.md
├── dataset
│   ├── aug_utils
│   │   ├── bbox_util.py
│   │   ├── data_aug.py
│   ├── custom
│   │   ├── images
│   │   ├── ssd_annotations
│   │   ├── temp_annotations
│   │   ├── xml2txt.py
│   │   └── yolo_annotations
│   ├── egohand
│   │   ├── egohands_data
│   │   │   ├── _LABELLED_SAMPLES
│   │   ├── mat2txt.py
│   │   ├── ssd_annotations
│   │   ├── temp_annotations
│   │   └── yolo_annotations
│   ├── raw_custom
│   │   ├── video
│   │   ├── image2label.py
│   │   └── video2image.py
│   ├── custom_augmentation.sh
│   ├── egohand_augmentation.sh
│   ├── augmentation.py
│   ├── split_data.py
│   ├── ssd_prepare.py 
│   └── yolo_prepare.py
├── modules
│   ├── handmodels.py
├── train
│   ├── train_ssd_mobilenetv2.py
│   ├── train_yolov4.ipynb
│   ├── train_yolov3.ipynb
├── test.py
├── demo.py
├── requirements.txt
├── README.md
└── venv
```

## How to test?

0. 환경 구축 
 0-1. virtualenv 환경 구축
 0-2. requirement.txt 설치

1. dataset 준비
 1-1. label 을 txt로 변환하기 
   * egohand dataset(.mat -> .txt)
   ```bash
   $ cd dataset/{dataset_name}
   $ python mat2txt.py
   ```
   * custom dataset(.xml -> .txt)
     * video -> image
     ```bash
     $ cd raw_custom
     $ python video2image.py
     ```

     * .xml label 생성 
      

     * .xml -> .txt
     ```bash
     $ cd dataset/{dataset_name}
     $ python xml2txt.py
     ```

     ```txt
     temp_annotations/train.txt , temp_annotations/test.txt 
      <image_name>,  <width>, <height>, <class>, <min_x>, <min_y>, <max_x>, <max_y>
      ```


 1-2. data augmentation
   ```bash
   # custom
   $ sh custom_augmentation.sh

   #egohand
   $ sh egohand_augmentation.sh
   ```

 1-3. train/ test data split (9:1)
  ```bash
   $ cd dataset/{dataset_name}
   $ python split_data.py
   ```

 1-4. yolo -> yolo 형식으로 변환
  ```bash
   $ cd dataset/{dataset_name}
   $ python yolo_prepare.py --name egohand
  ```

  ```txt
  [option]
  --name : egohand/ custom
  ```

  ```txt
  <class> <x_center> <y_center> <width> <height> (0~1 사이의 값으로 변환도 필요)
  ```

 1-5. ssd -> ssd 형식으로 변환 (.tfrecord , .pbtxt)
 ```bash
   $ cd dataset/{dataset_name}
   $ python ssd_prepare.py --name egohand
  ```
  ```txt
  [option]
  --name : egohand/ custom
  ```

  ```txt
  .csv -> .tfrecord, .pbtxt 생성
  <image_path>, <width>, <height>, <class>, <min_x>, <min_y>, <max_x>, <max_y>
  ```
 


2. train 
 2-1. yolo : darknet .ipynb 활용 
 2-2. ssd : tensosrflow api 활용 

3. test
  모듈로 embeded 함 
  ssd_mobilenetv1
  ssd_mobilenetv2
  yolov4-tiny
  yolov3-tiny
  google mediapipe palm detection


#### reference
* dataset

* data labeling

* data augmentation
  - https://github.com/Paperspace/DataAugmentationForObjectDetection  

* model
  - ssd_mobilenetv1
  https://github.com/victordibia/handtracking
  - ssd_mobilenetv2
  https://github.com/tensorflow/models
  - yolov3, yolov4
  https://github.com/AlexeyAB/darknet
  - google mediapipe palm detection
  https://github.com/google/mediapipe 
