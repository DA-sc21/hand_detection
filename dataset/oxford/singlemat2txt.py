import scipy.io as sio
import numpy as np
import os
import cv2
import argparse
import shutil
import glob

def bounding_box(xs, ys):
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    w = x_max - x_min
    h = y_max - y_min

    return (x_min + (w/2), y_min + (h/2), w, h)

def make_yolo_annotation(mode):
    base_path = mode+"_dataset/"
    dir = mode+"_data/"
    image_path_array = glob.glob(base_path +dir+"images/*.jpg")
    image_path_array.sort()
    # print(image_path_array)

    mat_path_array = glob.glob(base_path+dir+"annotations/*.mat")
    mat_path_array.sort()
    # print(mat_path_array)
    for mat_path in mat_path_array :
        annotation = sio.loadmat(mat_path)

        img_name = mat_path.split("/")[-1][:-4]
        img_path= base_path +dir+"images/"+img_name+".jpg"
        print(img_path)
        img = cv2.imread(img_path)

        width = np.size(img, 1)
        height =  np.size(img, 0)

        ann_txt=[]

        for boxes in annotation['boxes']:
            for box in boxes :
                box = list(box[0][0])
                bbox = [[i[0][0],i[0][1]] for i in box[:4]]
                # print(bbox)

                xs = [l[1] for l in bbox]
                ys = [l[0] for l in bbox]

                x,y,w,h = bounding_box(xs,ys)
                min_x = int(x-w/2)
                min_y = int(y-h/2)
                max_x = int(x+w/2)
                max_y = int(y+h/2)

                cv2.rectangle(img, (min_x, min_y),(max_x,max_y), (0, 255, 0), 2)
                ann_txt.append("%s %s %s %s %s" % (0, x / width, y / height, w / width, h / height))
        
        txt_path = img_path.split(".")[0]
        # print(ann_txt)
        cv2.imshow('with annotation ', img)
        cv2.waitKey(2)
        # cv2.waitKey(0)
        with open(txt_path+".txt","w") as txt_file :
            for ann in ann_txt :
                txt_file.write(ann+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --dataset_path hand_labels/manual_tests
    print("======================================")
    print("convert oxford dataset .mat annotation data to .txt yolo annotation data")
    print("======================================")
    parser.add_argument('--mode', default='test',type=str,help="train/test/validation")
    args = parser.parse_args()

    mode = args.mode

    make_yolo_annotation(mode)
    print("finish convert!!")