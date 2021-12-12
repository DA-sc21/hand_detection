import scipy.io as sio
import numpy as np
import cv2
import argparse
import glob

'''
python split_dataset.py --path augmentation_test_data --mode test
python split_dataset.py --path augmentation_train_data --mode train
'''
# data/obj/file_name.jpg
def split_dataset(path,mode):
    image_path_array = glob.glob(path+"/*.jpg")
    image_path_array.sort()
    #print(image_path_array)
    
    colab_path = "data/obj/"
    with open(path+"/"+mode+".txt","w") as txt_file :
        for img_path in image_path_array :
            txt_file.write(colab_path+img_path.split("/")[-1]+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --dataset_path hand_labels/manual_tests
    print("======================================")
    print("split train, test dataset")
    print("======================================")
    parser.add_argument('--path',type=str,help="dataset path")
    parser.add_argument('--mode', default='test',type=str,help="train/test")

    args = parser.parse_args()

    path = args.path
    mode = args.mode

    split_dataset(path,mode)
    print("finish split!!")