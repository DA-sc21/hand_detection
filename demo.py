import argparse
import glob
import os
import cv2

from model import Model


def test(opt):
    labels = {'hand' : 0}

    model = Model(opt)
    print('model input parameters : ', opt.ObjectDetection)
    
    files = sorted(glob.glob("%s/*.jpg" % opt.eval_data))

    model.forward(opt.ObjectDetection,files,True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    """ Model Architecture """
    parser.add_argument('--ObjectDetection', type=str, required=True, help='ObjectDetection stage. ssdmobilenetv1|ssdmobilenetv2 \
    yolov3-tiny|yolov4-tiny|mediapipe')
    parser.add_argument('--mode', type=str, default = '1_0_0',help='choose data mode of yolov4-tiny egohand : CMU : oxford = 1_0_0/2_1_2/4_1_4/8_1_4')

    opt = parser.parse_args()

    test(opt)
