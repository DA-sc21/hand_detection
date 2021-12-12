import numpy as np
import argparse
import random as rnd

'''
python make_dataset_txt.py --mode train --ratio all
python make_dataset_txt.py --mode test --ratio all
'''
def split_dataset(mode,ratio):
    egohand_path = "egohand/" + mode + ".txt"
    CMU_path = "CMUdataset/" + mode + ".txt"
    oxford_path = "oxford/" + mode + ".txt"
    aug_path = "augmentation_"+mode+"_data/"+mode+".txt"
    
    # define data num each dataset
    ego_num, CMU_num, oxford_num,aug_num = 0,0,0,0
    if ratio == "2_1_2":
        if mode == "train":
            ego_num, CMU_num, oxford_num = 2000,950,2000
        else :
            ego_num, CMU_num, oxford_num = 350,250,500
    elif ratio == "4_1_4":
        if mode == "train":
            ego_num, CMU_num, oxford_num = 4000,950,4000
        else :
            ego_num, CMU_num, oxford_num = 350,200,600  
    elif ratio == "8_1_4":
        if mode == "train":
            ego_num, CMU_num, oxford_num = 4000,500,2000
        else :
            ego_num, CMU_num, oxford_num = 350,100,170
    elif ratio=="all":
        if mode == "train":
            ego_num, CMU_num, oxford_num,aug_num = 4000,500,3000,25000
        else :
            ego_num, CMU_num, oxford_num,aug_num = 350,200,600,4000

    # get image file name for dataset path
    egohand_image = [line.strip() for line in open(egohand_path,"r").readlines()]
    CMU_image = [line.strip() for line in open(CMU_path,"r").readlines()]
    oxford_image = [line.strip() for line in open(oxford_path,"r").readlines()]
    aug_image = [line.strip() for line in open(aug_path,"r").readlines()]

    # random ratio for make new train/test split dataset
    new_dataset = []
    # print(len(oxford_image),oxford_num)
    new_dataset.extend(rnd.sample(egohand_image,ego_num))
    new_dataset.extend(rnd.sample(CMU_image,CMU_num))
    new_dataset.extend(rnd.sample(oxford_image,oxford_num))
    new_dataset.extend(rnd.sample(aug_image,aug_num))
    print(len(new_dataset))
    rnd.shuffle(new_dataset)

    # save dataset
    txt_path = mode + "_" + ratio
    with open(txt_path+".txt","w") as txt_file :
        for data in new_dataset :
            txt_file.write(data + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --dataset_path hand_labels/manual_tests
    print("======================================")
    print("split by ratio (EgoHands : CMU dataset : oxford)")
    print("======================================")
    parser.add_argument('--mode', default='test',type=str,help="train/test")
    parser.add_argument('--ratio', default='2_1_2',type=str,help="2_1_2/4_1_4/8_1_4/all")
    args = parser.parse_args()

    ratio = args.ratio
    mode = args.mode

    split_dataset(mode,ratio)
    print("finish split!!")