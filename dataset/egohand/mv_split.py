import shutil

## change!
f = open("train.txt","r")
while True:
    line = f.readline()
    if not line: break
    file_name = line.split("/")[-1].strip()
    print(file_name)
    source = "images/"+file_name
    # change!
    destination = "train/"+file_name
    shutil.move(source,destination)
f.close()

