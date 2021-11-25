import argparse
import csv

def csv_to_txt(opt):
    valid_path = opt.valid_path
    data = open(valid_path)
    reader = csv.reader(data)
    lines = list(reader)
    
    new_valid_path = "validation/"
    for line in lines[1:] :
        with open(new_valid_path+line[0][:-4]+".txt","a") as txt_file:
            txt_file.write("{c_name} {x_min} {y_min} {x_max} {y_max}\n".format(c_name=line[3],
            x_min=line[4],y_min =line[5], x_max = line[6], y_max = line[7]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid_path', type=str,help="validation data path")
    args = parser.parse_args()

    csv_to_txt(args)