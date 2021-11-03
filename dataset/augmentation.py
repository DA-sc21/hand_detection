# modify for our data set and make general code
# reference : https://github.com/Paperspace/DataAugmentationForObjectDetection/blob/master/test.py
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

    for image_path in image_path_array :
        img = cv2.imread(image_path)[:,:,::-1] 

        f = open(image_path[:-4]+".txt")
        line = f.readline()
        bboxes_temp= line.split()[1:]
        f.close()
        bboxes = []
        for box in bboxes_temp:
            box = box.split(',')
            b = [float(box[i]) for i in range(3,7)]
            b.append(0.0)
            bboxes.append(b)
            
        bboxes = np.array(bboxes)

        transforms = Sequence([RandomHorizontalFlip(1), 
        RandomHSV(hue=opt.hsv,saturation=opt.saturation, brightness=opt.brightness),
        RandomRotate(opt.rotation),
        RandomScale(opt.scaling, diff = True if opt.scaling>0 else False), 
        RandomTranslate(opt.translation)])
        img, bboxes = transforms(img, bboxes)


        cv2.imwrite("{path}_{num}.jpg".format(path=image_path[:-4],num=opt.n),img)
        with open("{path}_{num}.txt".format(path=image_path[:-4],num=opt.n),"w") as txt_file:
            file_name = image_path.split('/')[-1][:-4]
            txt_file.write("{name}_{num}.jpg ".format(name=file_name,num=opt.n))
            height,width = img.shape[:2]
            for line in bboxes:
                txt_file.write("{w},{h},{c},{x_min},{y_min},{x_max},{y_max} ".format(
                        w=width, h=height, c=int(line[-1]), x_min = line[0], y_min = line[1], x_max = line[2], y_max = line[3]
                ))
        plt.imshow(draw_rect(img, bboxes))
        plt.show()
        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_data', type=str, help='path to aug dataset')
    parser.add_argument('--n', type=int, help='times of same image(1,2,...)')

    # option
    parser.add_argument('--hsv', default = 0, type=int, help='hsv transform(0~179)')
    parser.add_argument('--saturation', default = 0, type=int, help='saturation transform(0~255)')
    parser.add_argument('--brightness', default = 0, type=int, help='brightness transform(0~100)')
    parser.add_argument('--rotation', default = 0, type=int, help='rotation 0 : false, 1 : true(0,90,180,270)')
    parser.add_argument('--scaling', default = 0, type=float, help='scaling')
    parser.add_argument('--translation', default = 0, type=float, help='translation')
    opt = parser.parse_args()

    augmentation(opt)