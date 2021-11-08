# train ssd_mobilenetv2 with tensorflow API (tf version 1 )

file structure
```
.
├── README.md
├── models-master (https://github.com/tensorflow/models)
│   ├── AUTHORS
│   ├── CODEOWNERS
│   ├── CONTRIBUTING.md
│   ├── ISSUES.md
│   ├── LICENSE
│   ├── README.md
│   ├── community
│   ├── official
│   ├── orbit
│   ├── research
│   └── tensorflow_models
├── requirements.txt
└── workspace
    ├── annotations
    │   └── label_map.pbtxt
    ├── egohands
    ├── egohands_data.zip
    ├── generate_tfrecord.py
    ├── images
    │   ├── test
    │   |     └──  test.tfrecord
    │   └── train
    │   |     └──  train.tfrecord
    ├── inference_graph
    ├── models
    │   ├── my_ssdmobilenetv2
    │   ├── ssd_mobilenet_v2_coco_2018_03_29
    │   └── ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    ├── prepare_dataset.py
    ├── tensorboard
    └── train.py
```

1. download or clone the repository 
(https://github.com/tensorflow/models)

2. set up environment
* Caution ! Please make a python virtual environment with version 3.6
(python 3.7 will conflict with python and numpy version)
  1. make virtual environment
  ```bash
  $ virtualenv --python=3.6 venv
  $ source venv/bin/activate
  ```

  2. install python package which was provided with tensorflow API
  ```bash
  $ cd models-master/research
  # Compile protos.
  $ protoc object_detection/protos/*.proto   --python_out=.
  # Install TensorFlow Object Detection API.
  $ cp object_detection/packages/tf1/setup.py .
  $ python -m pip install --use-feature=2020-resolver .
  ```

  3. test tensorflow API
  ```bash
  $ python object_detection/builders/model_builder_tf1_test.py
  ```

3. data prepare
  1. .mat -> .csv
  (reference : https://github.com/victordibia/handtracking)
  ```bash
  $ cd workspace
  $ python prepare_dataset.py
  ```
  2. .csv -> .tfrecord
  ```bash
  # train
  $ python generate_tfrecord.py --csv_input=images/train/train_labels.csv --output_path=images/train/train.record --image_dir=images/train
  # test
  $ python generate_tfrecord.py --csv_input=images/test/test_labels.csv --output_path=images/test/test.record --image_dir=images/test
  ```

  3. download pretraiend model and unzip to workspace/models

  4. make .pbtxt in workspace/annotation/label_map.pbtxt
  
  5. modify pipeline.config in pretrain model
  modify record path, label.pbtxt path, num_steps
  In the case of an memory error, you can modify batch_size to small size

4. train
  4-1. copy in models-master/research/object_detection/model_main.py to workspace/train.py
  ```bash
  python train.py --pipeline_config_path=models/my_ssdmobilenetv2/pipeline.config --model_dir=models/my_ssdmobilenetv2 --alsologtostderr
  ```

5. show tensorboard
  1. load tensorboard
  ```bash
  $ cd workspace/models/my_ssdmobilenetv2
  $ tensorboard --logdir=./
  ```

  2. go to localhost:6006