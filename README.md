# hand_detection

## analysis result

#### test mAP & inference time & memory 
##### front view (100 images-->200 TBD.)
<img src="imgs/front_view_result_100.png" width="70%">

##### side view (280 images)
<img src="imgs/side_view_result_280.png" width="70%">

#### validation mAP & inference time & memory (egohand evaluation data(400 images))
<img src="imgs/validation_result.png" width="70%">


#### 분석 
시간 관계상 open source에서 제공되는 ssdmobilenetv1은 모델 구조와 모델 가중치를 포함하는 .pb 를 그대로 사용하고, ssdmobilenetv2, yolov3-tiny-prn, yolov4-tiny에 대해서는 데이터를 수정하고, model config를 수정하여 직접 학습하였다.
  
validation set은 test data와 유사하기 때문에 높은 mAP를 보였지만, 자체 제작한 test 에서는 모두 성능이 떨어졌다.
그나마 general하게 작동하는 모델이 yolov4-tiny라는 것을 확인할 수 있다. 

yolov4의 경우, 기존의 YOLO에 Bag of freebies, Bag of specials 성능 향상 기법을 추가하고, universal한 feature를 추출하여 학습할 수 있도록 하였기 때문에 이와 같은 결과가 나왔을 것이라 생각한다. 
  
따라서 yolov4-tiny에 다양한 각도의 hand data를 추가하여 fine tuning을 진행해보고자 한다.

### structure
```
.
├── README.md
├── dataset
│   ├── aug_utils
│   │   ├── bbox_util.py
│   │   ├── data_aug.py
│   ├── augmentation.py
│   ├── custom
│   │   ├── front_view
│   │   ├── images
│   │   ├── side_view
│   │   ├── ssd_annotations
│   │   ├── test_annotations
│   │   ├── yolo_annotations
│   │   ├── images.txt
│   │   └── xml2txt.py
│   ├── custom_augmentation.sh
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
│   │   └── video2image.py
│   ├── egohand_augmentation.sh
│   ├── ssd_prepare.py 
│   ├── csv2txt.py 
│   ├── tests
│   ├── train
│   └── yolo_prepare.py
├── modules
│   ├── models
│   │   ├── ssdmobilenetv1
│   │   ├── ssdmobilenetv2
│   │   ├── yolov3-tiny
│   │   └── yolov4-tiny
│   ├── ssd_utils.py
│   └── yolo_utils.py
├── train
│   ├── ssdmobilenetv2
│   ├── yolov3-tiny
│   └── yolov4-tiny
├── valid_result
├── test_only_egohand
│   ├── test_result_front_view
│   └── test_result_side_view
├── demo.py
├── evaluate_mAP.py
├── model.py
├── test.py
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
 ```bash
 $ cd hand_detection/
 $ virtualenv --python=3.7 venv
 $ source venv/bin/activate
 ```
 0-2. requirements.txt 설치
 ```bash
 $ pip install -r requirements.txt
 ```

<br />

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
    https://github.com/tzutalin/labelImg tool 활용 

    * cf ) make validation set or test set for calculate mAP (.xml-> .txt)
<'class_name'> <'x_min'> <'y_min'> <'x_max'> <'y_max'>

     ```bash
     $ cd dataset/{dataset_name}
     $ python xml2txt.py --mode test
     ```

     ```txt
     test_annotations/train.txt , test_annotations/test.txt 
      <image_name>,  <width>, <height>, <class>, <min_x>, <min_y>, <max_x>, <max_y>
      ```

 
  1-3. yolo train dataset 준비
  ```txt
  <class> <x_center> <y_center> <width> <height> (0~1 사이의 값으로 변환도 필요)
  ```

 1-4. custom data ssd -> ssd 형식으로 변환 (.tfrecord , .pbtxt)
 ```bash
   $ cd train/ssdmobilenetv2/workspace/
   $ python generate_tfrecord.py
  ```

  ```txt
  .csv -> .tfrecord, .pbtxt 생성
  <image_path>, <width>, <height>, <class>, <min_x>, <min_y>, <max_x>, <max_y>
  ```

 
2. test or demo

* test : get inference time for sec/per image, get mAP
* demo : show detect result through opencv window
```bash
python test.py --eval_data dataset/{test dir}--ObjectDetection yolov4-tiny 
python demo.py --eval_data dataset/{test dir}--ObjectDetection yolov4-tiny 
```

``` txt
  모듈로 embeded 함 
  test : ssdmobilenetv1 | ssd_mobilenetv2 | yolov4-tiny | yolov3-tiny
  demo : ssdmobilenetv1 | ssd_mobilenetv2 | yolov4-tiny | yolov3-tiny | google mediapipe palm detection
  결과는 test/{model_name}에 저장되도록 한다.
```

<br />


## How to train with custom data & fine tuning?

* ssdmobilenetv2 : [how to train ssdmobilenetv2](train/ssdmobilenetv2)
* yolov3-tiny : [how to train yolov3-tiny](train/yolov3-tiny)
* yolov4-tiny : [how to train yolov4-tiny](train/yolov4-tinys)


#### reference
* dataset
  - http://vision.soic.indiana.edu/projects/egohands/
* data labeling
  - https://github.com/tzutalin/labelImg
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