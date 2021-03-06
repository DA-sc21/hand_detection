# modify https://github.com/victordibia/handtracking/blob/master/egohands_dataset_clean.py
import scipy.io as sio
import numpy as np
import os
import cv2
import shutil
import glob

def get_bbox_visualize(base_path, dir):
    image_path_array = []
    for root, dirs, filenames in os.walk(base_path + dir):
        for f in filenames:
            if(f.split(".")[1] == "jpg"):
                img_path = base_path + dir + "/" + f
                image_path_array.append(img_path)


    image_path_array.sort()
    boxes = sio.loadmat(
        base_path + dir + "/polygons.mat")

    polygons = boxes["polygons"][0]
    # first = polygons[0]
    # print(len(first))
    pointindex = 0

    for first in polygons:
        index = 0
        font = cv2.FONT_HERSHEY_SIMPLEX

        img_id = image_path_array[pointindex]
        img = cv2.imread(img_id)
        
        img_params = {}
        img_params["width"] = np.size(img, 1)
        img_params["height"] = np.size(img, 0)
        head, tail = os.path.split(img_id)
        img_params["filename"] = tail
        img_params["path"] = os.path.abspath(img_id)
        img_params["type"] = "train"
        pointindex += 1

        boxarray = []
        label_arr = []
        for pointlist in first:
            pst = np.empty((0, 2), int)
            max_x = max_y = min_x = min_y = height = width = 0

            findex = 0
            for point in pointlist:
                if(len(point) == 2):
                    x = int(point[0])
                    y = int(point[1])

                    if(findex == 0):
                        min_x = x
                        min_y = y
                    findex += 1
                    max_x = x if (x > max_x) else max_x
                    min_x = x if (x < min_x) else min_x
                    max_y = y if (y > max_y) else max_y
                    min_y = y if (y < min_y) else min_y
                    # print(index, "====", len(point))
                    appeno = np.array([[x, y]])
                    pst = np.append(pst, appeno, axis=0)
                    cv2.putText(img, ".", (x, y), font, 0.7,
                                (255, 255, 255), 2, cv2.LINE_AA)

            hold = {}
            hold['minx'] = min_x
            hold['miny'] = min_y
            hold['maxx'] = max_x
            hold['maxy'] = max_y
            if (min_x > 0 and min_y > 0 and max_x > 0 and max_y > 0):
                boxarray.append(hold)
                labelrow = [tail, np.size(img, 1), np.size(img, 0),0,min_x, min_y, max_x, max_y]
                label_arr.append(labelrow)

            cv2.polylines(img, [pst], True, (0, 255, 255), 1)
            cv2.rectangle(img, (min_x, max_y),
                          (max_x, min_y), (0, 255, 0), 1)


        txt_path = img_id.split(".")[0]
        if not os.path.exists(txt_path + ".txt"):
            cv2.putText(img, "DIR : " + dir + " - " + tail, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
            cv2.imshow('Verifying annotation ', img)
            tmp_text_file = ""
            flag = 1
            with open(txt_path+".txt","w") as txt_file:
                for line in label_arr:
                    if line[0] == tmp_text_file :
                        txt_file.write(",".join([str(a) for a in line[1:]])+" ")
                    else :
                        tmp_text_file = line[0]
                        txt_file.write("%s "%line[0]+ ",".join([str(a) for a in line[1:]])+" ")
                        if flag :
                            flag = 0
                        else :
                            txt_file.write("\n")
                      
                                       
            print("===== saving txt file for ", tail)
        cv2.waitKey(2)  # close window when a key press is detected

def generate_txt_files(image_dir):
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            get_bbox_visualize(image_dir, dir)
            filepaths = glob.glob(root+dir+"/*.txt") + glob.glob(root+dir+"/*.jpg")
            for filepath in filepaths:
                shutil.move(filepath,'images')
    
    os.rmdir('egohands_data')

    print("label generation complete!\n")



def rename_files(image_dir):
    print("Renaming files")
    loop_index = 0
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            for f in os.listdir(image_dir + dir):
                if (dir not in f):
                    if(f.split(".")[1] == "jpg"):
                        loop_index += 1
                        os.rename(image_dir + dir +
                                  "/" + f, image_dir + dir +
                                  "/" + dir + "_" + f)
                else:
                    break
    print("finish renaming!")
    generate_txt_files("egohands_data/_LABELLED_SAMPLES/")



rename_files("egohands_data/_LABELLED_SAMPLES/")
