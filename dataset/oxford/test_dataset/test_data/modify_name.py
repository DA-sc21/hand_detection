import os
 
def changeName(path):
    for filename in os.listdir(path):
        print(path+filename, '=>', path+"test_"+filename)
        os.rename(path+filename, path+"test_"+filename)
 
changeName('images/')