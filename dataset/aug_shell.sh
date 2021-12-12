 # !/bin/bash
 # EgoHands Done
python augmentation.py --aug_data egohand/train --n 1 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05 --option train
python augmentation.py --aug_data egohand/test --n 1 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05 --option test

# CMU Done
python augmentation.py --aug_data CMUdataset/train --n 1 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05 --option train
python augmentation.py --aug_data CMUdataset/test --n 1 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05 --option test

 # Oxford Done
python augmentation.py --aug_data oxford/test_dataset/test_data/images --n 1 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05 --option test
python augmentation.py --aug_data oxford/training_dataset/training_data/images --n 1 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05 --option train
python augmentation.py --aug_data oxford/validation_dataset/validation_data/images --n 1 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05 --option train


 # EgoHands Done
python augmentation.py --aug_data egohand/train --n 2 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05 --option train
python augmentation.py --aug_data egohand/test --n 2 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05 --option test

# CMU Done
python augmentation.py --aug_data CMUdataset/train --n 2 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05 --option train
python augmentation.py --aug_data CMUdataset/test --n 2 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05 --option test

 # Oxford Done
python augmentation.py --aug_data oxford/test_dataset/test_data/images --n 2 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05 --option test
python augmentation.py --aug_data oxford/training_dataset/training_data/images --n 2 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05 --option train
python augmentation.py --aug_data oxford/validation_dataset/validation_data/images --n 2 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05 --option train


 # EgoHands Done
python augmentation.py --aug_data egohand/train --n 3 --brightness 50 --rotation 1 --option train
python augmentation.py --aug_data egohand/test --n 3 --brightness 50 --rotation 1 --option test

# CMU DONE
python augmentation.py --aug_data CMUdataset/train --n 3 --brightness 50 --rotation 1 --option train
python augmentation.py --aug_data CMUdataset/test --n 3 --brightness 50 --rotation 1 --option test

 # Oxford DONE
python augmentation.py --aug_data oxford/test_dataset/test_data/images --n 3 --brightness 50 --rotation 1 --option test
python augmentation.py --aug_data oxford/training_dataset/training_data/images --n 3 --brightness 50 --rotation 1 --option train
python augmentation.py --aug_data oxford/validation_dataset/validation_data/images --n 3 --brightness 50 --rotation 1 --option train
