# modify https://github.com/cansik/yolo-hand-detection/issues/11

import glob
import os
import json
import argparse
import shutil

from PIL import Image


def load(file):
    with open(file, 'r') as f:
        return json.load(f)

def write(file, content):
    with open(file, 'w') as outfile:
        outfile.write(content)

def bounding_box(xs, ys):
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    w = x_max - x_min
    h = y_max - y_min

    return (x_min + (w/2), y_min + (h/2), w, h)

def json2txt(ann_map,dataset,opt):
    for key in ann_map:
        anns = ann_map[key]
        default_file = os.path.basename(ann_map[key][0])
        default_file_name = os.path.splitext(default_file)[0]

        im = Image.open("%s/%s.jpg" % (dataset, default_file_name))
        width = im.size[0]
        height = im.size[1]


        ann_txt = []

        for annFile in anns:
            filename = os.path.basename(annFile)
            name = os.path.splitext(filename)[0]
            detections = []
            print(annFile)
            desc = load(annFile)
            keypoints = desc["hand_pts"]

            xs = [l[0] for l in keypoints]
            ys = [l[1] for l in keypoints]
        
            x, y, w, h = bounding_box(xs, ys)

            ann_txt.append("%s %s %s %s %s" % (0, x / width, y / height, w / width, h / height))

        write("%s/%s.txt" % (opt, key), '\n'.join(ann_txt))
        shutil.copyfile("%s/%s.jpg" % (dataset, default_file_name), "%s/%s.jpg" % (opt, key))
        data.append("%s.jpg" % (key))



    write("%s.txt" % opt, '\n'.join(data))
    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --dataset_path hand_labels/manual_test
    parser.add_argument('--dataset_path', type=str,help="dataset path")
    parser.add_argument('--mode', default='test',type=str,help="train/test")
    args = parser.parse_args()

    dataset = args.dataset_path

    data = []
    annotations = glob.glob("%s/*.json" % dataset)

    # create annotation dict
    ann_map = {}
    for ann in annotations:
        key = os.path.splitext(os.path.basename(ann))[0][:-2]
        # not one person
        if len(key.split("_")) == 2 :
            key = key.split("_")[0]
    
        if not key in ann_map:
            ann_map[key] = []

        ann_map[key].append(ann)
    # print(ann_map)
    json2txt(ann_map,dataset,args.mode)