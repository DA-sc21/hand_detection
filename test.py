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

    model.forward(opt.ObjectDetection,files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    # parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    # parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    # parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Model Architecture """
    parser.add_argument('--ObjectDetection', type=str, required=True, help='ObjectDetection stage. ssdMobileNetv1|ssdMobileNetv2 \
    yolov3-tiny|yolov4-tiny|mediapipe')
    
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=416,
                        help='the number of output channel of Feature extractor')
    opt = parser.parse_args()

    test(opt)
