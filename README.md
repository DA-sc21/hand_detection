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
│   │   ├── egohands_data (will be removed)
│   │   │   ├── _LABELLED_SAMPLES
│   │   ├── images
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

### custom dataset 
* situation
- each 2 people, 2 view points(옆, 앞)
  - 깍지손 (10초)
  - 필기 중인 손
  - 

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
     raw_custom에 video 추가하기 
     ```bash
     $ cd raw_custom
     $ python video2image.py --video_dir videos
     ```

     * .xml label 생성 
    https://github.com/tzutalin/labelImg 참고하여 작성

     * .xml -> .txt
     ```bash
     $ cd dataset/{dataset_name}
     $ python xml2txt.py
     ```

     ```txt
     temp_annotations/train.txt , temp_annotations/test.txt 
      <image_name>,  <width>, <height>, <class>, <min_x>, <min_y>, <max_x>, <max_y>
      ```
 1-2. train/ test data split (9:1) of custom data
  ```bash
   $ cd dataset/{dataset_name}
   $ python split_data.py
   ```

 1-3. custom data yolo -> yolo 형식으로 변환
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

 1-4. custom data ssd -> ssd 형식으로 변환 (.tfrecord , .pbtxt)
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
 
2. test
```bash
python test.py --eval_data dataset/test --ObjectDetection yolov4-tiny 
```
  모듈로 embeded 함 
  ssd_mobilenetv1
  ssd_mobilenetv2
  yolov4-tiny
  yolov3-tiny
  google mediapipe palm detection
  결과는 test/{model_name}에 저장되도록 한다.

3. mAP 측정
다음과 같은 형태로 txt를 바꾸어준다.
```
ground_truth의 txt
dataset/test/{image_name}.txt 들의 형태 
<class_name> <left> <top> <right> <bottom> [<difficult>]

detection-result의 txt
<class_name> <confidence> <left> <top> <right> <bottom>
```

```bash
$ cd test
$ python evaluate_mAP.py --model {model_name}
```


## How to train with custom data & fine tuning?

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
     raw_custom에 video 추가하기 
     ```bash
     $ cd raw_custom
     $ python video2image.py --video_dir videos
     ```

     * .xml label 생성 
    https://github.com/tzutalin/labelImg 참고하여 작성

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
 2-1. {최종 Model} 에 대해 cfg 여러 개 작성
   - fine tuning
     - dataset split random ()
     - optimizer
     - freeze layer? (TBD)
 2-2. training

3. test
  각각 다른 cfg에 대해 test 수행


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
  https://github.com/cansik/yolo-hand-detection 
  - google mediapipe palm detection
  https://github.com/google/mediapipe 

* evaluate mAP
    https://github.com/Cartucho/mAP