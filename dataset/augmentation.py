# modify https://github.com/Paperspace/DataAugmentationForObjectDetection/blob/master/test.py
from aug_utils.data_aug import *
from aug_utils.bbox_util import *
import cv2 
import pickle as pkl
import numpy as np 
import matplotlib.pyplot as plt
import glob
import argparse

def augmentation(opt):
    image_path_array = glob.glob(opt.aug_data+"/*.jpg")
    # print(image_path_array)
    image_path_array.sort()

    img = cv2.imread(image_path_array[0])[:,:,::-1] 

    f = open(image_path_array[0][:-4]+".txt")
    line = f.readline()
    bboxes_temp= line.split()[1:]
    bboxes = []
    for box in bboxes_temp:
        box = box.split(',')
        b = [float(box[i]) for i in range(3,7)]
        b.append(0.0)
        bboxes.append(b)
    bboxes = np.array(bboxes)
    transforms = Sequence([RandomHorizontalFlip(1), RandomScale(0.2, diff = True), RandomRotate(10)])
    img, bboxes = transforms(img, bboxes)

    plt.imshow(draw_rect(img, bboxes))
    plt.show()
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_data', type=str, help='path to aug dataset')
    parser.add_argument('-n', type=int, help='times of same image(1,2,...)')

    # option
    parser.add_argument('--scaling', type=str, help='scaling')
    parser.add_argument('--translation', type=str, help='translation')
    parser.add_argument('--rotation', type=str, help='rotation')
    parser.add_argument('--resizing', type=str, help='resizing')
    parser.add_argument('--hsv', type=str, help='hsv transform')
    opt = parser.parse_args()

    augmentation(opt)