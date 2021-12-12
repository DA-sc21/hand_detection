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
        print(image_path)
        img = cv2.imread(image_path)[:,:,::-1] 
        height, width = img.shape[:2]

        f = open(image_path[:-4]+".txt")
        bboxes = []
        while True:
            line = f.readline()
            if not line: break
            box = [float(i) for i in line.split()]
            x_center,y_center,w,h = box[1:]
            x_min = x_center*width-(w*width//2)
            y_min = y_center*height-(h*height//2)
            x_max = x_center*width +(w*width//2)
            y_max = y_center*height+(h*height//2)
            b = [x_min,y_min,x_max,y_max]
            b.append(0.0)
            bboxes.append(b)

        f.close()   

        bboxes = np.array(bboxes)
        # print(bboxes)

        transforms = Sequence([RandomHorizontalFlip(1), 
        RandomHSV(hue=opt.hsv,saturation=opt.saturation, brightness=opt.brightness),
        RandomRotate(opt.rotation),
        RandomScale(opt.scaling, diff = True if opt.scaling!=0 else False), 
        RandomTranslate(opt.translation)])
        img, bboxes = transforms(img, bboxes)

        new_path = "augmentation_"+opt.option+"_data/" + image_path[:-4].split("/")[-1]
        cv2.imwrite("{path}_{num}.jpg".format(path=new_path,num=opt.n),img)
        with open("{path}_{num}.txt".format(path=new_path,num=opt.n),"w") as txt_file:
            for line in bboxes:
                dw = 1./width
                dh = 1./height
                x = (int(line[0])+int(line[2]))/2.0
                y = (int(line[1])+int(line[3]))/2.0
                w=int(line[2])-int(line[0])
                h=int(line[3])-int(line[1])
                x *= dw
                w *= dw
                y *= dh
                h *= dh
                txt_file.write("{c} {x} {y} {w} {h}\n".format(
                        c=0, x = x, y = y, w = w, h = h
                ))
        # plt.imshow(draw_rect(img, bboxes))
        # plt.show()
        # break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_data', type=str, help='path to aug dataset')
    parser.add_argument('--n', type=int, help='times of same image(1,2,...)')
    parser.add_argument('--option', type=str, help='train/test')

    # option
    parser.add_argument('--hsv', default = 0, type=int, help='hsv transform(0~179)')
    parser.add_argument('--saturation', default = 0, type=int, help='saturation transform(0~255)')
    parser.add_argument('--brightness', default = 0, type=int, help='brightness transform(0~100)')
    parser.add_argument('--rotation', default = 0, type=int, help='rotation 0 : false, 1 : true(0,90,180,270)')
    parser.add_argument('--scaling', default = 0.1, type=float, help='scaling')
    parser.add_argument('--translation', default = 0.01, type=float, help='translation')
    opt = parser.parse_args()

    augmentation(opt)