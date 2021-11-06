# https://bblib.net/entry/convert-voc-to-yolo-xml-to-yolo-xml-to-txt

import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
import argparse

dirs = ['images']
classes = ['hand']


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

# <class> <x_center> <y_center> <width> <height> (0~1 사이의 값으로 변환도 필요)
def convert_annotation_yolo(dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(dir_path + '/' + basename_no_ext + '.xml' ,encoding='UTF8')
    out_file = open(output_path + basename_no_ext + '.txt', 'w' ,encoding='UTF8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))

        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

# width, height, <class>, min_x, min_y, max_x, max_y
def convert_annotation_ssd(dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(dir_path + '/' + basename_no_ext + '.xml' ,encoding='UTF8')
    out_file = open(output_path + basename_no_ext + '.txt', 'w' ,encoding='UTF8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = str(size.find('width').text)
    h = str(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))

        out_file.write(w + " " + h + " " + str(cls_id) + " " + " ".join([str(a) for a in b]) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='yolo|ssd')

    opt = parser.parse_args()   
    cwd = getcwd()

    for dir_path in dirs:
        full_dir_path = cwd + '/' + dir_path 

        output_path = cwd + "/ssd_annotations/"
        if opt.model == 'yolo':
            output_path = cwd + "/yolo_annotations/"
    
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        image_paths = glob.glob(dir_path + '/*.xml')
        print(full_dir_path)
        list_file = open(full_dir_path + '.txt', 'w')

        for image_path in image_paths:
            list_file.write(image_path + '\n')
            if opt.model == 'yolo':
                convert_annotation_yolo(full_dir_path, output_path, image_path)
            elif opt.model == 'ssd':
                convert_annotation_ssd(full_dir_path, output_path, image_path)

        list_file.close()

        print("Finished processing: " + dir_path)    