import cv2
import argparse
import glob

def video2image(opt):
    video_paths = glob.glob(opt.video_dir+"/*.mov")
    video_paths += glob.glob(opt.video_dir+"/*.mp4")
    print(video_paths)
    # video_paths = ['/Users/yejoonko/git/Project/Capstone/hand_detection/dataset/raw_custom/videos/view_front_person_1_writing.mov']
    count = int(opt.start_num)
    for vp in video_paths:
        video_name = vp.split("/")[-1][:-4]
        vidcap = cv2.VideoCapture(vp)
        success,image = vidcap.read()
        new_path = '../custom/images/'
        while success:
            if count % 15 == 0:
                image = cv2.resize(image,(1280,720))
                cv2.imwrite(new_path+video_name+"_%04d.jpg" % (count//15), image)     # save frame as JPEG file
            success,image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1

        print("finish! convert video to frame {name}".format(name=video_name))
    print(count)
    print("all convert finish!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, help='path to video directory')
    parser.add_argument('--start_num',type=str,help='start_num')
    opt = parser.parse_args()

    video2image(opt)