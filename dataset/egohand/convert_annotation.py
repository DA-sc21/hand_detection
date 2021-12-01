import numpy as np
import os
import argparse

def parse_args():

    desc = "convert annotation image structure"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--file_path', type=str, default="annotation/train/train_labels.txt", help='file_path')
    return parser.parse_args()


def change_annotation(file_path):
    new_file_path = "annotation/new_train/"    
    if file_path == "annotation/test/test_labels.txt" :
        new_file_path = "annotation/new_test/"

    with open(file_path, 'r') as file:
        reader = [ line.strip().split(' ') for line in file.readlines()]

    for line in reader:
        name = line[0][:-3]
        with open(new_file_path+name+"txt","w") as txt_file:
            for i in range(1,len(line)):
                dw = 1./1280
                dh = 1./720
                tmp = line[i].split(',')
                x = (int(tmp[0])+int(tmp[2]))/2.0
                y = (int(tmp[1])+int(tmp[3]))/2.0
                w=int(tmp[2])-int(tmp[0])
                h=int(tmp[3])-int(tmp[1])
                x *= dw
                w *= dw
                y *= dh
                h *= dh

                txt_file.write("0 {x} {y} {w} {h}\n".format(x=x,y=y,w=w,h=h))

                       


if __name__ == '__main__':
    args = parse_args()
    change_annotation(args.file_path)