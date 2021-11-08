import os
import random

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Split data, copy to train/test folders
def split_data_test_eval_train(image_dir):
    create_directory("images")
    create_directory("images/train")
    create_directory("images/test")


    data_size = 4000
    loop_index = 0
    data_sampsize = int(0.1 * data_size)
    test_samp_array = random.sample(range(data_size), k=data_sampsize)

    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            for f in os.listdir(image_dir + dir):
                if(f.split(".")[1] == "jpg"):
                    loop_index += 1
                    print(loop_index, f)

                    if loop_index in test_samp_array:
                        os.rename(image_dir + dir +
                                  "/" + f, "images/test/" + f)
                        os.rename(image_dir + dir +
                                  "/" + f.split(".")[0] + ".txt", "images/test/" + f.split(".")[0] + ".txt")
                    else:
                        os.rename(image_dir + dir +
                                  "/" + f, "images/train/" + f)
                        os.rename(image_dir + dir +
                                  "/" + f.split(".")[0] + ".txt", "images/train/" + f.split(".")[0] + ".txt")
                    print(loop_index, image_dir + f)
            print(">   done scanning director ", dir)
            os.remove(image_dir + dir + "/polygons.mat")
            os.rmdir(image_dir + dir)

        print("Train/test content generation complete!")
        generate_label_files("images/")