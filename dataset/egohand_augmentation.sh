# !/bin/bash
# change color
python augmentation.py --aug_data egohand/images --n 1 --hsv 179 --saturation 30 --brightness 50

# change rotation
python augmentation.py --aug_data egohand/images --n 2 --rotation 1

# change scaling
python augmentation.py --aug_data egohand/images --n 3 --scaling 0.2

# change translation
python augmentation.py --aug_data egohand/images --n 4 --translation 0.05

# change all things (x 3 times)
python augmentation.py --aug_data egohand/images --n 5 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05
python augmentation.py --aug_data egohand/images --n 6 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05
python augmentation.py --aug_data egohand/images --n 7 --hsv 179 --saturation 30 --brightness 50 --rotation 1 --scaling 0.2 --translation 0.05