# hand_detection

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
   $ cd dataset/{dataset_name}
   $ python augmentation.py
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