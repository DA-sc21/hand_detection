import scipy.io as sio
import numpy as np
import cv2
import argparse
import glob

# data/obj/file_name.jpg
def split_dataset(mode):
    image_path_array = []
    if mode == 'test' :
        base_path = mode+"_dataset/"
        dir = mode+"_data/"
        image_path_array = glob.glob(base_path +dir+"images/*.jpg")
        image_path_array.sort()
    else :
        base_path = mode+"_dataset/"
        base_path2 ="validation_dataset/"
        dir = mode+"_data/"
        dir2 = "validation_data/"
        image_path_array = glob.glob(base_path +dir+"images/*.jpg")
        image_path_array.extend(glob.glob(base_path2 +dir2+"images/*.jpg"))
        image_path_array.sort()

    #print(image_path_array)
    
    colab_path = "data/obj/"
    with open(mode+".txt","w") as txt_file :
        for img_path in image_path_array :
            txt_file.write(colab_path+img_path.split("/")[-1]+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --dataset_path hand_labels/manual_tests
    print("======================================")
    print("split train, test dataset")
    print("======================================")
    parser.add_argument('--mode', default='test',type=str,help="training/test")

    args = parser.parse_args()

    mode = args.mode

    split_dataset(mode)
    print("finish split!!")